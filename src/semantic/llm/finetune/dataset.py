import os.path
import random
from tqdm import tqdm
from typing import List, Tuple
from dataclasses import dataclass
from transformers import DataCollatorWithPadding

from augmentation import random_crop, crop_fixed_length, adjust_volume, add_gaussian_noise
from datasets import Audio, load_dataset, concatenate_datasets
from torch.utils.data import Dataset

import sys
sys.path.append("/opt/nikhil.kothari/all/mbed/stage1/bert_retro_mae")

from arguments import DataArguments


class TrainDatasetForAudioEmbedding(Dataset):
    def __init__(self, args: DataArguments):
        self.augmentaion_dataset = load_dataset(
            "PolyAI/minds14", name="en-AU", split="train"
        )
        self.augmentaion_dataset = self.augmentaion_dataset.cast_column(
            "audio", Audio(sampling_rate=args.sampling_rate)
        )
        self.n_augmentation_samples = len(self.augmentaion_dataset)

        if type(args.train_data_dir_list) is list:
            train_datasets = []

            for directory in args.train_data_dir_list:
                temp_dataset = load_dataset(
                    "audiofolder", data_dir=directory, split="train"
                )

                train_datasets.append(temp_dataset)
            self.dataset = concatenate_datasets(train_datasets)

        elif os.path.isdir(args.train_data_dir_list):
            self.dataset = load_dataset(
                "audiofolder", data_dir=args.train_data_dir_list, split="train"
            )

        else:
            raise ValueError(
                "args.train_data_dir_list should either be a directory or a list of directories"
            )

        self.args = args

        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=args.sampling_rate)
        )
        self.dataset = [
            self.split_audio_sample(
                datapoint, args.passage_max_len, overlap_duration=args.overlap_len
            )
            for datapoint in tqdm(self.dataset)
        ]
        self.dataset = [datapoint for data in self.dataset for datapoint in data]

        self.total_len = len(self.dataset)
        self.sampling_rate = args.sampling_rate
        self.passage_max_len = args.passage_max_len
        self.query_max_len = args.query_max_len

    def split_audio_sample(self, sample, segment_duration, overlap_duration=0):
        """
        Splits an audio sample into smaller segments.

        Args:
            sample (dict): A dictionary containing the audio path, array, and sampling rate.
            segment_duration (int): Desired duration of each segment in seconds.
            overlap_duration (int): Overlap between consecutive segments in seconds.

        Returns:
            list: A list of dictionaries containing segmented audio data.
        """
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]

        segment_length = segment_duration * sample_rate
        overlap_length = overlap_duration * sample_rate

        total_length = len(audio_array)
        step_size = segment_length - overlap_length
        segments = []

        for start_idx in range(0, total_length, step_size):
            end_idx = start_idx + segment_length
            if end_idx > total_length:
                break  # Stop if the last segment exceeds the total length

            segment_array = audio_array[start_idx:end_idx]
            segments.append(
                {
                    "audio": {
                        "path": sample["audio"]["path"],
                        "array": segment_array,
                        "sampling_rate": sample_rate,
                    }
                }
            )

        return segments

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        passage = self.dataset[item]
        query = random_crop(passage["audio"]["array"], self.sampling_rate, self.query_max_len)

        # get some audio sample to augment query sample
        random_idx = random.randint(0, self.n_augmentation_samples)
        aug_sample = crop_fixed_length(array=self.augmentaion_dataset[random_idx]["audio"]["array"], 
                                       sample_rate=self.sampling_rate, 
                                       sample_length=len(query)/self.sampling_rate, 
                                       fit_audio_to_sample_length=True)
        aug_sample = adjust_volume(aug_sample, volume_range=(0.6, 0.8))

        # get some gaussian noise in the query
        query = add_gaussian_noise(query, noise_level=0.02)

        passage["audio"]["query_array"] = query + aug_sample
        return passage


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    query_max_len: int = 10
    passage_max_len: int = 30

    def __call__(self, features):
        query = [f["query_array"] for f in features]
        passage = [f["array"] for f in features]

        q_collated = self.tokenizer(
            query, 
            sampling_rate=self.tokenizer.sampling_rate,
            max_length=self.query_max_len,
            return_tensors="pt", 
            truncation=True,
            padding=True).input_features

        p_collated = self.tokenizer(
            passage,
            sampling_rate=self.tokenizer.sampling_rate,
            max_length=self.passage_max_len,
            return_tensors="pt", 
            truncation=True,
            padding=True).input_features
        return {"query": q_collated, "passage": p_collated}
