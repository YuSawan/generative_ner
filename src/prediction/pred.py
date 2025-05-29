from typing import Any

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel

import wandb
from src.data.preprocessor import Preprocessor
from src.gpt.base.utils import regex


def _generate(
        model_inputs: list[list[dict[str, Any]]],
        model: PreTrainedModel,
        preprocessor: Preprocessor,
        max_new_tokens: int
    ) -> list[str]:
    prompts = preprocessor.tokenizer.apply_chat_template(model_inputs, tokenize = False, add_generation_prompt=True)
    tokenized_chats = preprocessor.tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    tokenized_chats = tokenized_chats.to(model.device)

    generated_tokens = model.generate(**tokenized_chats, max_new_tokens=max_new_tokens, pad_token_id=preprocessor.tokenizer.eos_token_id)
    generated_texts: list[str] = []
    for tokens in generated_tokens:
        generated_text = preprocessor.tokenizer.decode(tokens).replace(preprocessor.tokenizer.eos_token, "\n")
        generated_text = generated_text.split(preprocessor.response_template)[-1].strip()
        generated_texts.append(generated_text)
    return generated_text

@torch.no_grad()
def predict(
        model: PreTrainedModel,
        predict_dataset: Dataset,
        preprocessor: Preprocessor,
        names2labels: dict[str, str],
        batch_size: int = 1,
        max_new_tokens: int = 512,
    ) -> list[dict[str, Any]]:
    format = preprocessor.format
    pbar = tqdm(total=len(predict_dataset), desc="Predict")
    predictions = []
    for document in predict_dataset:
        pbar.update(1)

        texts, model_inputs, gold_spans = [], [], []
        for example in document["examples"]:
            eid = example["id"]
            text = example["text"]
            if format in ['collective', 'universal']:
                texts.append(text)
                gold_spans.append([(ent["start"], ent["end"], ent["label"]) for ent in example["entities"]])
                model_inputs.append([messages[:-1] for messages in preprocessor.get_messages([example])][0])
                if len(texts) == batch_size:
                    generated_texts = _generate(model_inputs, model, preprocessor, max_new_tokens)
                    for t, gs, gt in zip(texts, gold_spans, generated_texts):
                        ps = []
                        preds = preprocessor.parse_output(gt, format)
                        for p in sorted(set(preds)):
                            if ": " not in p:
                                continue
                            label, mention = p.split(": ")[:2]
                            try:
                                ps.extend([(s, e, names2labels[label]) for s, e in regex(t.lower(), mention)])
                            except KeyError:
                                ps.extend([(s, e, label) for s, e in regex(t.lower(), mention)])
                        predictions.append({"id": eid, "text": t, "golds": gs, "preds": ps, 'generated_text': gt})

            elif format == 'individual':
                for label, messages in zip(names2labels.values(), preprocessor.get_messages([example])):
                    texts.append(text)
                    gold_spans.append([(ent["start"], ent["end"], ent["label"]) for ent in example["entities"] if ent["label"] == label])
                    model_inputs.append(messages[:-1])
                    if len(texts) == batch_size:
                        generated_texts = _generate(model_inputs, model, preprocessor, max_new_tokens)
                        for t, gs, gt in zip(texts, gold_spans, generated_texts):
                            ps = []
                            preds = preprocessor.parse_output(gt, format)

                            for p in sorted(set(preds)):
                                try:
                                    ps.extend([(s, e, names2labels[label]) for s, e in regex(text.lower(), p)])
                                except KeyError:
                                    ps.extend([(s, e, label) for s, e in regex(text.lower(), p)])
                            predictions.append({"id": eid, "text": t, "golds": gs, "preds": ps, 'generated_text': gt})
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
