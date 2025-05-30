
import os

import torch
from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)
from transformers.trainer_utils import set_seed
from trl import DataCollatorForCompletionOnlyLM, setup_chat_format

from src import DatasetArguments, ModelArguments, parse_args
from src.data import Preprocessor, get_splits
from src.evaluation import evaluate, submit_wandb_evaluate
from src.prediction import convert_predictions_to_json, predict, submit_wandb_predict
from src.training import LoggerCallback, setup_logger

logger = logging.get_logger(__name__)
TOKEN = os.environ.get('TOKEN', True)


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"data_args: {data_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    data_files = {k: getattr(data_args, f"{k}_file") for k in ["train", "validation", "test"]}
    data_files = {k: v for k, v in data_files.items() if v is not None}
    cache_dir = model_args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    if not data_args.labels2names:
        label_set = set()
        for document in raw_datasets["train"]:
            for example in document["examples"]:
                for entity in example["entities"]:
                    label_set.add(entity["label"])
        data_args.labels2names = {label: label for label in sorted(label_set)}
    logger.info(f"labels: {data_args.labels2names}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16,
        use_cache = False,
        device_map = "auto"
    )
    if not tokenizer.chat_template:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    preprocessor = Preprocessor(
        tokenizer,
        data_args.labels2names,
        language=data_args.language,
        format=data_args.format,
        system_message=data_args.system_prompt,
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=preprocessor.instruction_template,
        response_template=preprocessor.response_template
    )

    if model_args.prev_path:
        model = PeftModel.from_pretrained(model, model_args.prev_path)
        if not training_args.do_train:
            model = model.merge_and_unload()
            model = model.to(training_args.device)
    else:
        if training_args.do_train:
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
        else:
            model = model.to(training_args.device)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        trainer = Trainer(
            model,
            args=training_args,
            train_dataset=splits['train'],
            eval_dataset=splits['validation'],
            data_collator=collator,
        )
        trainer.add_callback(LoggerCallback(logger))
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

        if training_args.do_eval or training_args.do_predict:
            model = model.merge_and_unload()

    if training_args.do_eval:
        names2labels = {v: k for k, v in data_args.labels2names.items()}
        predictions = predict(model, raw_datasets['test'], preprocessor, names2labels, training_args.eval_batch_size)
        metrics = {f"eval_{k}": v for k, v in evaluate(predictions).items()}
        logger.info(f"eval metrics: {metrics}")

        if training_args.do_train:
            trainer.log_metrics("eval", metrics)
            submit_wandb_evaluate(metrics)
            submit_wandb_predict(predictions)
            if training_args.save_strategy != "no":
                trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        names2labels = {v: k for k, v in data_args.labels2names.items()}
        predictions = predict(model, raw_datasets["validation"], preprocessor, names2labels, training_args.eval_batch_size)
        outputs_data = convert_predictions_to_json(predictions, raw_datasets["validation"])
        outputs_data.to_json(os.path.join(training_args.output_dir, "predictions.jsonl"))

def cli_main() -> None:
    data_args, model_args, training_args = parse_args()
    if data_args.validation_file is None:
        training_args.eval_strategy = "no"
    main(data_args, model_args, training_args)


if __name__ == '__main__':
    cli_main()
