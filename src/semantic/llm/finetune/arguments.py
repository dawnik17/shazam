import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataArguments:
    train_data_dir_list: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)
    sampling_rate: int = field(default=16_000)

    query_max_len: int = field(
        default=10,
        metadata={
            "help": "The maximum length for the query audio (in seconds). Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=30,
        metadata={
            "help": "The maximum length for the passage or main audio (in seconds). Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    overlap_len: int = field(
        default=5,
        metadata={
            "help": "While splitting the main audio into n segments of 'passage_max_len' each"
            "What should be the overlap (in seconds)."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000,
        metadata={"help": "the max number of examples for each dataset"},
    )

    query_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    # def __post_init__(self):
    #     if not os.path.exists(self.train_data):
    #         raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(
        default=False, metadata={"help": "share negatives across devices"}
    )
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(
        default="cls", metadata={"help": "the pooling method, should be cls or mean"}
    )
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(
        default=True, metadata={"help": "use passages in the same batch as negatives"}
    )