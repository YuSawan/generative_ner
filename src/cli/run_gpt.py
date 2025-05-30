import logging
import os
import random
import sys
from typing import Optional

from datasets import Dataset, load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from tqdm.auto import tqdm

import wandb
from src import DatasetArguments, GptModelArguments, parse_args_gpt
from src.data.preprocessor import Example, Preprocessor
from src.evaluation import evaluate, submit_wandb_evaluate
from src.gpt import DEFAULT_TIMEOUT, BatchAPI_Wrapper, CostChecker, OpenAI_APIWrapper
from src.gpt.base.utils import regex
from src.prediction import submit_wandb_predict

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
APITOKEN_FILE = os.environ['APIFILE']


def sample_demonstration(dataset: Dataset, k: int, seed: int) -> list[Example]:
    random.seed(seed)
    examples = []
    for example in dataset["examples"]:
        examples.extend([e for e in example])
    random.shuffle(examples)
    return examples[:k] if len(examples) >= k else examples


def convert_examples_to_messages(
        example: Example,
        format: str,
        labels2names: dict[str, str],
        language: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
    text, entities = example["text"], example["entities"]
    if format == 'collective':
        return Preprocessor.get_collective_prompt(text, entities, labels2names, language, system_prompt)
    elif format == 'universal':
        return Preprocessor.get_universal_prompt(text, entities, labels2names, language, system_prompt)
    elif format == 'individual':
        return Preprocessor.get_individual_prompt(text, entities, labels2names, language, system_prompt)
    else:
        raise NotImplementedError(f"Format '{format}' is not implemented.")


def main_gpt(data_args: DatasetArguments, model_args: GptModelArguments) -> None:
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

    logger.info(f"data_args: {data_args}")
    logger.info(f"model_args: {model_args}")

    openai_api = OpenAI_APIWrapper(model_name=model_args.model_name, api_file=APITOKEN_FILE, timeout=DEFAULT_TIMEOUT)
    cost_checker = CostChecker(
        model_name = model_args.model_name,
        cost_usd_limit = model_args.total_cost_limit,
        use_batchapi= True if model_args.mode == 'batch' else False,
        estimate=True if model_args.mode=='estimate' else False,
    )
    if model_args.mode == 'batch':
        batch_wrapper = BatchAPI_Wrapper(APITOKEN_FILE, model_args.output_dir)
        try:
            batch_jobs = batch_wrapper.get_batch_id()
            if batch_jobs:
                raise FileExistsError("Batch job already exists. Please delete the existing batch job before submitting a new one.")
        except FileNotFoundError:
            pass
    else:
        if model_args.mode == 'generate':
            wandb.init(project="gpt-ner", name=f"{model_args.model_name}")
            wandb.log(vars(data_args))
            wandb.log(vars(model_args))
        predictions = []

    if raw_datasets["validation"]:
        sampled_demo = sample_demonstration(raw_datasets["validation"], model_args.k, model_args.seed)
    else:
        sampled_demo = sample_demonstration(raw_datasets["train"], model_args.k, model_args.seed)

    demonstrations = []
    for example in sampled_demo:
        messages = convert_examples_to_messages(example, data_args.format, data_args.labels2names, data_args.language)
        demonstrations.append(messages)

    pbar = tqdm(total=len(raw_datasets["test"]))
    for data in raw_datasets["test"]:
        pbar.update()
        for example in data["examples"]:
            text = example["text"]
            messages = convert_examples_to_messages(example, data_args.format, data_args.labels2names, data_args.language, data_args.system_prompt)
            if data_args.format in ['collective', 'universal']:
                demo_messages = []
                for demo in demonstrations:
                    demo_messages.extend(demo)

                gold_output = messages[-1]['content']
                messages = messages[:1] + demo_messages + messages[1:-1] if data_args.system_prompt else demo_messages + messages[:-1]
                gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"]]

                if model_args.mode in ['generate', 'debug']:
                    results = openai_api.generate(
                        messages,
                        temperature=model_args.temperature,
                        max_token_length=model_args.max_token_length,
                        top_p=model_args.top_p,
                        seed=model_args.seed,
                        n=model_args.n,
                        debug=True if model_args.mode == 'debug' else False
                    )
                    generated_text = results['choices'][0]['message']['content'] # TODO: Handle model_args.n choices
                    if model_args.mode == 'debug':
                        print("Instruction:")
                        for m in messages:
                            print(f'{m["role"]}: ', m["content"])
                        print("-----------------------")
                        print("Gold:\n"+gold_output)
                        print("Generated:\n"+generated_text)

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
                    fee = cost_checker(results)
                else:
                    if model_args.mode == "batch":
                        openai_api.append_messages_to_batch(
                            messages,
                            temperature=model_args.temperature,
                            max_token_length=model_args.max_token_length,
                            top_p=model_args.top_p,
                            seed=model_args.seed,
                            n=model_args.n,
                        )
                    estimated_results = openai_api.estimate(messages, gold_output)
                    fee = cost_checker(estimated_results)
                logger.info(fee)
            elif data_args.format == 'individual':
                labels = names2labels.keys()
                system_message = messages[:1] if data_args.system_prompt else []
                model_input = messages[1:3] if data_args.system_prompt else messages[:2]
                label_model_input = messages[3:] if data_args.system_prompt else messages[2:]
                demo_input = [d[:3] for d in demonstrations] if data_args.system_prompt else [d[:2] for d in demonstrations]
                demo_label_input = [d[3:] for d in demonstrations] if data_args.system_prompt else [d[2:] for d in demonstrations]
                for i, label in enumerate(labels):
                    gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"] if ent["label"] == label]
                    gold_output = label_model_input[i*2+1]['content']
                    demo_messages = []
                    for di, dl in zip(demo_input, demo_label_input):
                        demo_messages.extend(di)
                        demo_messages.extend(dl[i*2: i*2+2])
                    messages = system_message + demo_messages + model_input + label_model_input[i*2: i*2+1]

                    if model_args.mode in ['generate', 'debug']:
                        results = openai_api.generate(
                            messages,
                            temperature=model_args.temperature,
                            max_token_length=model_args.max_token_length,
                            top_p=model_args.top_p,
                            seed=model_args.seed,
                            n=model_args.n,
                            debug=True if model_args.mode == 'debug' else False
                        )

                        generated_text = results['choices'][0]['message']['content'] # TODO: Handle model_args.n choices
                        if model_args.mode == 'debug':
                            print("Instruction:")
                            for m in messages:
                                print(f'{m["role"]}: ', m["content"])
                            print("-----------------------")
                            print("Gold:\n"+gold_output)
                            print("Generated:\n"+generated_text)

                        pred_spans = []
                        preds = Preprocessor.parse_output(generated_text)
                        for p in sorted(set(preds)):
                            if not isinstance(p, str):
                                continue
                            pred_spans.extend([(s, e, label) for s, e in regex(text.lower(), p)])
                        predictions.append({"id": example["id"], "text": text, "golds": gold_spans, "preds": pred_spans, 'generated_text': generated_text})
                        fee = cost_checker(results)
                    else:
                        if model_args.mode == "batch":
                            openai_api.append_messages_to_batch(
                                messages,
                                temperature=model_args.temperature,
                                max_token_length=model_args.max_token_length,
                                top_p=model_args.top_p,
                                seed=model_args.seed,
                                n=model_args.n,
                            )
                        estimated_results = openai_api.estimate(messages, gold_output)
                        fee = cost_checker(estimated_results)
                    logger.info(fee)
            else:
                raise NotImplementedError(f"Format '{data_args.format}' is not implemented.")

        if model_args.mode == 'debug':
            logger.info("Debug mode is enabled. Exiting after the first example.")
            break

    pbar.close()
    cost_checker.total()

    if model_args.mode in ['generate', 'debug']:
        metrics = evaluate(predictions)
        if model_args.mode == 'generate':
            submit_wandb_evaluate(metrics)
            submit_wandb_predict(predictions)
        else:
            for key, value in metrics.items():
                logger.info(f"{key}: {value}")

    if model_args.mode == 'batch':
        openai_api.save_batch(save_dir=model_args.output_dir, ensure_ascii=False)
        batch_wrapper.submit_batch(description="Eval Test")


def cli_main_gpt() -> None:
    data_args, model_args = parse_args_gpt()
    os.makedirs(model_args.output_dir, exist_ok=True)
    main_gpt(data_args, model_args)


if __name__ == '__main__':
    cli_main_gpt()
