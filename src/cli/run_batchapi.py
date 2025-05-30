import os

from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from tqdm.auto import tqdm

import wandb
from src import DatasetArguments, GptModelArguments, parse_args_gpt
from src.data import Preprocessor
from src.evaluation import evaluate, submit_wandb_evaluate
from src.gpt import BatchAPI_Wrapper, CostChecker
from src.gpt.base.utils import regex
from src.prediction import submit_wandb_predict

APITOKEN_FILE = os.environ['APIFILE']

def check_batch_job_status(batch_wrapper: BatchAPI_Wrapper) -> bool:
    batch_jobs = batch_wrapper.get_batch_id()
    for batch_job in batch_jobs:
        status = batch_wrapper.check_job_status(batch_job["batch_job_id"])
        if status != "completed":
            return False
    return True


def main_batchapi(data_args: DatasetArguments, model_args: GptModelArguments) -> None:
    batch_wrapper = BatchAPI_Wrapper(APITOKEN_FILE, model_args.output_dir)
    cost_checker = CostChecker(model_name=model_args.model_name, cost_usd_limit=model_args.total_cost_limit, estimate=True, use_batchapi=True)
    if not check_batch_job_status(batch_wrapper):
        raise RuntimeError("Some batch jobs are still running or not started yet. Please wait until they are completed.")

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
    names2labels = {v: k for k, v in data_args.labels2names.items()}

    wandb.init(project="gpt-ner", name=f"{model_args.model_name}")
    wandb.log(vars(data_args))
    wandb.log(vars(model_args))

    batch_jobs = batch_wrapper.get_batch_id()
    all_generations = []
    for batch_job in batch_jobs:
        model_inputs, results = batch_wrapper.load_batch(batch_job)
        for model_input, result in zip(model_inputs, results):
            _result = result['response']['body']
            generated_text = _result['choices'][0]['message']["content"]
            all_generations.append(generated_text)
            _ = cost_checker(_result)

    predictions = []
    if data_args.format in ['collective', 'universal']:
        pbar = tqdm(total=len(raw_datasets["test"]))
        for data in raw_datasets["test"]:
            pbar.update(1)
            for example in data["examples"]:
                text = example["text"]
                gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"]]
                generated_text = all_generations.pop(0)
                pred_spans = []
                preds = Preprocessor.parse_output(generated_text)
                for p in sorted(set(preds)):
                    if not isinstance(p, tuple) or len(p) != 2:
                        continue
                    mention, label = p[0], p[1]
                    try:
                        pred_spans.extend([(s, e, names2labels[label]) for s, e in regex(text.lower(), mention)])
                    except KeyError:
                        pred_spans.extend([(s, e, label) for s, e in regex(text.lower(), mention)])
                predictions.append({"id": example["id"], "text": text, "golds": gold_spans, "preds": pred_spans, 'generated_text': generated_text})
        pbar.close()
    elif data_args.format == 'individual':
        pbar = tqdm(total=len(raw_datasets["test"]))
        for data in raw_datasets["test"]:
            pbar.update(1)
            for example in data["examples"]:
                text = example["text"]
                for label in names2labels.values():
                    gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"] if ent["label"] == label]
                    generated_text = all_generations.pop(0)

                    pred_spans = []
                    preds = Preprocessor.parse_output(generated_text)
                    for p in sorted(set(preds)):
                        if not isinstance(p, str):
                            continue
                        pred_spans.extend([(s, e, label) for s, e in regex(text.lower(), p)])
                    predictions.append({"id": example["id"], "text": text, "golds": gold_spans, "preds": pred_spans, 'generated_text': generated_text})
        pbar.close()
    else:
        raise NotImplementedError(f"Format {data_args.format} is not implemented.")

    cost_checker.total()
    metrics = evaluate(predictions)
    submit_wandb_evaluate(metrics)
    submit_wandb_predict(predictions)


def cli_main_batchapi() -> None:
    data_args, model_args = parse_args_gpt()
    if not os.path.exists(model_args.output_dir):
        raise FileNotFoundError(f"Output directory {model_args.output_dir} does not exist. Please create it before running the script.")
    os.makedirs(model_args.output_dir, exist_ok=True)
    main_batchapi(data_args, model_args)


if __name__ == '__main__':
    cli_main_batchapi()
