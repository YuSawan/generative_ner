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
    format: str = 'collective' # "collective", "individual" or "universal"
    labels2names: Optional[dict[str, str]] = None
    system_prompt: Optional[str] = None


@dataclass
class ModelArguments:
    """Model arguments."""
    model_name: str
    model_max_length: int
    cache_dir: Optional[str] = None
    prev_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class GptModelArguments:
    """Model arguments for GPT."""
    model_name: str
    total_cost_limit: float
    top_p: float
    temperature: float
    seed: int
    k: int
    n: int
    max_token_length: int
    mode: str # "generate", "estimate", "debug", or "batch"
    output_dir: str
    cache_dir: Optional[str] = None


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


def parse_args_gpt() -> tuple[DatasetArguments, GptModelArguments]:
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", metavar="FILE", required=True,
        help="Path to the configuration file in YAML format."
    )
    parser.add_argument(
        "--output_dir", "-o", metavar="DIR", required=True,
        help="Directory to save the output files."
    )
    parser.add_argument(
        "--mode", "-m", metavar="MODE", default="estimate",
        choices=["generate", "estimate", "debug", "batch"],
        help="Mode of operation: 'generate' for generating outputs, 'estimate' for cost estimation, 'debug' for debugging, or 'batch' for batch processing."
    )

    args, _ = parser.parse_known_args()
    config = vars(load_config_as_namespace(args.config_file))

    data_config = config.pop("dataset")
    model_config = config.pop("gpt_model")

    model_config["output_dir"] = args.output_dir
    model_config["mode"] = args.mode

    arguments = DatasetArguments(**data_config)
    model_args = GptModelArguments(**model_config)

    return arguments, model_args
