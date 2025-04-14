from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging,
)
from transformers.trainer_utils import set_seed
from trl import DataCollatorForCompletionOnlyLM

from eval import evaluate
from pred import predict
from preprocessor import Preprocessor, data_preprocess
from training_utils import setup_logger

logger = logging.get_logger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    language: str = 'en' # "en" or "ja"
    format: str = 'single' # "single" or "multi"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    checkpoint_path: Optional[str] = None
    cache_dir: Optional[str] = 'tmp/'
    extend_context: bool = False


def finetune(args: Arguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    data_files = {k: getattr(args, f"{k}_file") for k in ["train", "validation", "test"]}
    data_files = {k: v for k, v in data_files.items() if v is not None}
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    label_set = set()
    for document in raw_datasets["train"]:
        for example in document["examples"]:
            for entity in example["entities"]:
                label_set.add(entity["label"])
    labels = sorted(label_set)
    logger.info(f"labels: {labels}")


    tokenizer = AutoTokenizer.from_pretrained(args.model, token=True)
    preprocessor = Preprocessor(tokenizer, labels, language=args.language, extend_context=args.extend_context, format=args.format)

    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer = tokenizer,
        response_template = preprocessor.response_template
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype = torch.bfloat16,
        use_cache = False,
        device_map = "auto"
    )

    if training_args.do_train:
        if training_args:
            with training_args.main_process_first(desc="dataset map pre-processing"):
                train_dataset = data_preprocess(preprocessor, raw_datasets['train']) if raw_datasets['train'] else None
                validation_dataset = data_preprocess(preprocessor, raw_datasets['validation']) if raw_datasets['validation'] else None
        else:
            train_dataset = data_preprocess(preprocessor, raw_datasets['train']) if raw_datasets['train'] else None
            validation_dataset = data_preprocess(preprocessor, raw_datasets['validation']) if raw_datasets['validation'] else None

        peft_config = LoraConfig(
            r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        trainer = Trainer(
            model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
            args = training_args,
            tokenizer = tokenizer
        )
        result = trainer.train()
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)
    else:
        model = PeftModel.from_pretrained(model, args.output_dir)


    if training_args.do_eval:
        model.eval()
        result = evaluate(model, raw_datasets['test'], preprocessor)
        print(result)

    if training_args.do_predict:
        model.eval()
        predicts = predict(model, raw_datasets['test'], preprocessor)
        for p in predicts:
            print(p['text'])
            print(p['gold'])
            print(p['pred'])


if __name__ == '__main__':
    CONFIG_FILE = Path(__file__).parents[1] / "default.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    set_seed(training_args.seed)
    finetune(args, training_args)

