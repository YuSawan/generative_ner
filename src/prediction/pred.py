from typing import Any

import torch
from datasets import Dataset
from peft import PeftModelForCausalLM
from tqdm.auto import tqdm

import wandb
from src.data.preprocessor import Preprocessor
from src.gpt.base.utils import regex


def _generate(messages: list[dict[str, Any]], model: PeftModelForCausalLM, preprocessor: Preprocessor) -> str:
    model_input = messages[:-1]
    tokenized_chat = preprocessor.tokenizer.apply_chat_template(
        model_input,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    tokenized_chat = tokenized_chat.to(model.device)
    generated_tokens = model.generate(tokenized_chat)
    generated_text = preprocessor.tokenizer.decode(generated_tokens[0]).replace(preprocessor.tokenizer.eos_token, "\n")
    generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return generated_text

@torch.no_grad()
def predict(
        model: PeftModelForCausalLM,
        predict_dataset: Dataset,
        preprocessor: Preprocessor,
        names2labels: dict[str, str],
    ) -> list[dict[str, Any]]:
    format = preprocessor.format
    pbar = tqdm(total=len(predict_dataset), desc="Predict")
    predictions = []
    for document in predict_dataset:
        pbar.update(1)
        for example in document["examples"]:
            eid = example["id"]
            text = example["text"]
            if format in ['collective', 'universal']:
                gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"]]
                for messages in preprocessor.get_messages([example]):
                    generated_text = _generate(messages, model, preprocessor)
                    pred_spans = []
                    preds = preprocessor.parse_output(generated_text, format)
                    for p in sorted(set(preds)):
                        if ": " not in p:
                            continue
                        label, mention = p.split(": ")
                        try:
                            pred_spans.extend([(s, e, names2labels[label]) for s, e in regex(text.lower(), mention)])
                        except KeyError:
                            pred_spans.extend([(s, e, label) for s, e in regex(text.lower(), mention)])
                    predictions.append({"id": eid, "text": text, "golds": gold_spans, "preds": pred_spans, 'generated_text': generated_text})

            elif format == 'individual':
                for label, messages in zip(names2labels.values(), preprocessor.get_messages([example])):
                    gold_spans = [(ent["start"], ent["end"], ent["label"]) for ent in example["entities"] if ent["label"] == label]
                    generated_text = _generate(messages, model, preprocessor)
                    pred_spans = []
                    preds = preprocessor.parse_output(generated_text, format)
                    for p in sorted(set(preds)):
                        try:
                            pred_spans.extend([(s, e, names2labels[label]) for s, e in regex(text.lower(), p)])
                        except KeyError:
                            pred_spans.extend([(s, e, label) for s, e in regex(text.lower(), p)])
                    predictions.append({"id": eid, "text": text, "golds": gold_spans, "preds": pred_spans, 'generated_text': generated_text})
            else:
                raise NotImplementedError(f"Format {format} is not implemented")

    return predictions


def convert_predictions_to_json(predictions: list[dict[str, Any]], dataset: Dataset) -> Dataset:
    results: dict[str, list[dict[str, int|str]]] = {}
    for prediction in predictions:
        try:
            results[prediction["id"]].extend([{"start": s, "end": e, "label": label} for s, e, label in prediction["preds"]])
        except KeyError:
            results[prediction["id"]] = [{"start": s, "end": e, "label": label} for s, e, label in prediction["preds"]]

    for data in dataset:
        for example in data["examples"]:
            example["prediction"] = results[example["id"]]

    return dataset


def submit_wandb_predict(predictions: list[dict[str, Any]]) -> None:
    columns = ["id", "text", "gold", "predictions", "generated_text"]
    result_table = wandb.Table(columns=columns)

    for prediction in predictions:
        golds = sorted(set([f"{label}: {prediction['text'][s:e]}" for s, e, label in prediction["golds"]]))
        preds = sorted(set([f"{label}: {prediction['text'][s:e]}" for s, e, label in prediction["preds"]]))

        result_table.add_data(
            prediction["id"],
            prediction["text"],
            ', '.join(golds),
            ', '.join(preds),
            prediction["generated_text"]
        )
    wandb.log({"predictions": result_table})
