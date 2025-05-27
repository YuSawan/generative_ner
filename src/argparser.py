import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from typing import Optional

import yaml
from transformers import HfArgumentParser, TrainingArguments


def load_config_as_namespace(config_file: str | os.PathLike) -> Namespace:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)


@dataclass
class DatasetArguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    language: str = 'en' # "en" or "ja"
    format: str = 'single' # "single" or "multi"


@dataclass
class ModelArguments:
    """Model arguments."""
    model_name: str
    model_max_length: int
    cache_dir: Optional[str] = None
    prev_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


def parse_args() -> tuple[DatasetArguments, ModelArguments, TrainingArguments]:
    parser = ArgumentParser()
    hfparser = HfArgumentParser(TrainingArguments)
    parser.add_argument(
        "--config_file", metavar="FILE", required=True
    )

    args, extras = parser.parse_known_args()
    config = vars(load_config_as_namespace(args.config_file))
    training_args = hfparser.parse_args_into_dataclasses(extras)[0]

    data_config = config.pop("dataset")
    model_config = config.pop("model")

    arguments = DatasetArguments(**data_config)
    model_args = ModelArguments(**model_config)
    training_args = replace(training_args, **config)

    return arguments, model_args, training_args
