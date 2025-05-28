import logging
import os
import random
import sys
from typing import Iterator, Union

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
        examples: list[Example],
        format: str,
        labels2names: dict[str, str],
        language: str,
        system_prompt: str | None,
    ) -> Iterator[Union[list[dict[str, str]], dict[str, list[dict[str, str]]]]]:
    for text, entities in Preprocessor.segment(examples):
        base_messages = Preprocessor.get_base_prompt(text, system_prompt, language)
        if format == 'collective':
            messages = Preprocessor.get_collective_prompt(entities, labels2names, language)
            yield base_messages + messages
        elif format == 'universal':
            messages = Preprocessor.get_universal_prompt(entities, labels2names, language)
            yield base_messages + messages
        elif format == 'individual':
            label_messages: dict[str, list[dict[str, str]]] = {}
            for label, message in zip(labels2names.keys(), Preprocessor.get_individual_prompt(entities, labels2names, language)):
                label_messages[label] = base_messages + message
            yield label_messages
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

    if data_args.format != 'individual':
        demonstrations: list[dict[str, str]] = []
        for s in convert_examples_to_messages(sampled_demo, data_args.format, data_args.labels2names, data_args.language, data_args.system_prompt):
            assert isinstance(s, list)
            demonstrations.extend(s[1:])
    else:
        label_demonstrations: dict[str, list[dict[str, str]]] = {label: [] for label in data_args.labels2names.keys()}
        for ls in convert_examples_to_messages(sampled_demo, data_args.format, data_args.labels2names, data_args.language, data_args.system_prompt):
            assert isinstance(ls, dict)
            for label, messages in ls.items():
                label_demonstrations[label].extend(messages[1:])

    pbar = tqdm(total=len(raw_datasets["test"]))
    for data in raw_datasets["test"]:
        pbar.update()
        for example in data["examples"]:
            text = example["text"]
            if data_args.format in ['collective', 'universal']:
                messages = [_ for _ in convert_examples_to_messages([example], data_args.format, data_args.labels2names, data_args.language, data_args.system_prompt)][0]
                gold_output = messages[-1]['content']
                messages = messages[:1] + demonstrations + messages[1:-1]
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
                        for m in messages[-3:]:
                            print(f'{m["role"]}: ', m["content"])
                        print("-----------------------")
                        print("Gold:\n"+gold_output)
                        print("Generated:\n"+generated_text)

                    pred_spans = []
                    preds = Preprocessor.parse_output(generated_text, data_args.format)
                    for p in sorted(set(preds)):
                        if ": " not in p:
                            continue
                        label, mention = p.split(": ")
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
                assert label_demonstrations
                label_messages = [_ for _ in convert_examples_to_messages([example], data_args.format, data_args.labels2names, data_args.language, data_args.system_prompt)][0]
                for label, messages in label_messages.items():
                    gold_output = messages[-1]['content']
                    messages = messages[:1] + label_demonstrations[label] + messages[1:-1]
                    gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"] if ent["label"] == label]

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
                            for m in messages[-3:]:
                                print(f'{m["role"]}: ', m["content"])
                            print("-----------------------")
                            print("Gold:\n"+gold_output)
                            print("Generated:\n"+generated_text)

                        pred_spans = []
                        preds = Preprocessor.parse_output(generated_text, data_args.format)
                        for p in sorted(set(preds)):
                            try:
                                pred_spans.extend([(s, e, names2labels[label]) for s, e in regex(text.lower(), p)])
                            except KeyError:
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
