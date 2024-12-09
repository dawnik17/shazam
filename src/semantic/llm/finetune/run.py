"""
rm run.log && torchrun --nproc_per_node 8 -m run --experiment_name v1 >> run.log
"""

import logging
import os
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Dict

from transformers import WhisperProcessor
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

import sys

sys.path.append("/opt/nikhil.kothari/all/mbed/stage2/flag_emb_finetune/")

from utils import TrainingConfig
from arguments import (
    ModelArguments,
    DataArguments,
    RetrieverTrainingArguments as TrainingArguments,
)
from data import TrainDatasetForAudioEmbedding, EmbedCollator
from modeling import BiEncoderModel
from trainer import BiTrainer


logger = logging.getLogger(__name__)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    experiment_name: str = field(
        metadata={"help": "experiment name"},
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "local rank of process"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, ScriptArguments))
    model_args, data_args, script_args = parser.parse_args_into_dataclasses()

    # local_rank = script_args.local_rank
    experiment_name = script_args.experiment_name

    # Read Config
    experiment_config_path = f"/home/nikhil.kothari/all/mbed/stage2/flag_emb_finetune/config/experiment/{experiment_name}.yml"
    config = TrainingConfig.from_file(file_path=experiment_config_path)

    # Set Output Directory
    output_dir = os.path.join(
        config.output_dir,
        f"{config.base_model_path.rstrip('/').split('/')[-1]}-finetune-emb-experiment-{experiment_name}",
    )

    model_args: ModelArguments
    data_args: DataArguments

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        # deepspeed=config.deepspeed,
        lr_scheduler_type=getattr(config, "lr_scheduler", "cosine"),
        report_to=["tensorboard"],
        weight_decay=config.weight_decay,
        warmup_steps=getattr(config, "warmup_steps", -1),
        do_train=True,
        resume_from_checkpoint=config.resume_from_checkpoint,
        logging_steps=1,
        negatives_cross_device=config.negatives_cross_device,
        temperature=config.temperature,
        fix_position_embedding=config.fix_position_embedding,
        sentence_pooling_method=config.sentence_pooling_method,
        normlized=config.normlized,
        use_inbatch_neg=config.use_inbatch_neg,
    )

    # Model Arguments
    model_args.model_name_or_path = config.base_model_path

    # Dataset Arguments
    data_args.train_data = config.dataset_name_or_path
    data_args.train_group_size = config.train_group_size
    data_args.query_max_len = config.query_max_len
    data_args.passage_max_len = config.passage_max_len
    data_args.max_example_num_per_dataset = config.max_example_num_per_dataset
    data_args.query_instruction_for_retrieval = config.query_instruction_for_retrieval
    data_args.passage_instruction_for_retrieval = (
        config.passage_instruction_for_retrieval
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = WhisperProcessor.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        # cache_dir=model_args.cache_dir,
        # use_fast=False,
    )

    # num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     (
    #         model_args.config_name
    #         if model_args.config_name
    #         else model_args.model_name_or_path
    #     ),
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    # )
    # logger.info("Config: %s", config)

    model = BiEncoderModel(
        model_name=model_args.model_name_or_path,
        normlized=training_args.normlized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_inbatch_neg=training_args.use_inbatch_neg,
    )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForAudioEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
        ),
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
