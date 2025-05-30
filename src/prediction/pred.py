from typing import Any, Optional

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
    return generated_texts


def convert_text_to_spans(
        format: str,
        ids: list[str],
        texts: list[str],
        gold_spans: list[list[tuple[int, int, str]]],
        generated_texts: list[str],
        preprocessor: Preprocessor,
        names2labels: dict[str, str],
        labels: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
    predictions = []
    if format in ['collective', 'universal']:
        for t, gs, gt, eid in zip(texts, gold_spans, generated_texts, ids):
            ps = []
            preds = preprocessor.parse_output(gt)
            for p in sorted(set(preds)):
                if not isinstance(p, tuple) or len(p) != 2:
                    continue
                mention, label = p[0], p[1]
                try:
                    ps.extend([(s, e, names2labels[label]) for s, e in regex(t.lower(), mention)])
                except KeyError:
                    ps.extend([(s, e, label) for s, e in regex(t.lower(), mention)])
            predictions.append({"id": eid, "text": t, "golds": gs, "preds": ps, 'generated_text': gt})
    elif format == 'individual':
        assert labels is not None and len(labels) == len(generated_texts)
        for t, gs, gt, lb, eid in zip(texts, gold_spans, generated_texts, labels, ids):
            ps = []
            preds = preprocessor.parse_output(gt)
            for p in sorted(set(preds)):
                if not isinstance(p, str):
                    continue
                ps.extend([(s, e, lb) for s, e in regex(t.lower(), p)])
            predictions.append({"id": eid, "text": t, "golds": gs, "preds": ps, 'generated_text': gt})
    else:
        raise NotImplementedError(f"Format {format} is not implemented")

    return predictions


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

        ids, texts, model_inputs, gold_spans = [], [], [], []
        for example in document["examples"]:
            eid = example["id"]
            text = example["text"]
            entities = example["entities"]
            messages = preprocessor.get_messages(text, entities)
            if format in ['collective', 'universal']:
                ids.append(eid)
                texts.append(text)
                gold_spans.append([(ent["start"], ent["end"], ent["label"]) for ent in entities])
                model_inputs.append(messages[:-1])
                if len(texts) == batch_size:
                    generated_texts = _generate(model_inputs, model, preprocessor, max_new_tokens)
                    predictions.extend(convert_text_to_spans(
                        format, ids, texts, gold_spans, generated_texts, preprocessor, names2labels
                    ))
                    ids, texts, model_inputs, gold_spans = [], [], [], []

            elif format == 'individual':
                labels = []
                model_input = messages[:3] if preprocessor.system_message else messages[:2]
                label_model_input = messages[3:] if preprocessor.system_message else messages[2:]
                for i, label in enumerate(names2labels.values()):
                    ids.append(eid)
                    texts.append(text)
                    labels.append(label)
                    gold_spans.append([(ent["start"], ent["end"], ent["label"]) for ent in example["entities"] if ent["label"] == label])
                    model_inputs.append(model_input + [label_model_input[i*2]])
                    if len(texts) == batch_size:
                        generated_texts = _generate(model_inputs, model, preprocessor, max_new_tokens)
                        predictions.extend(convert_text_to_spans(
                            format, ids, texts, gold_spans, generated_texts, preprocessor, names2labels, labels
                        ))
                        texts, model_inputs, gold_spans, ids, labels = [], [], [], [], []
            else:
                raise NotImplementedError(f"Format {format} is not implemented")

        if texts and model_inputs and gold_spans:
            generated_texts = _generate(model_inputs, model, preprocessor, max_new_tokens)
            predictions.extend(convert_text_to_spans(
                format, ids, texts, gold_spans, generated_texts, preprocessor, names2labels, None if format in ['collective', 'universal'] else labels
            ))

    return predictions


def convert_predictions_to_json(predictions: list[dict[str, Any]], dataset: Dataset) -> list[dict[str, Any]]:
    results: dict[str, list[dict[str, int|str]]] = {}
    for prediction in predictions:
        try:
            results[prediction["id"]].extend([{"start": s, "end": e, "label": label} for s, e, label in prediction["preds"]])
        except KeyError:
            results[prediction["id"]] = [{"start": s, "end": e, "label": label} for s, e, label in prediction["preds"]]

    jsondata = []
    for data in dataset:
        result = {"id": data["id"], "examples": []}
        for example in data["examples"]:
            data = {k: v for k, v in example.items()}
            data.update({"predictions": results[example["id"]]})
            result["examples"].append(data)
        jsondata.append(result)

    return jsondata


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
