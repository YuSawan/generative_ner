from typing import Any

import torch
import wandb
from datasets import Dataset
from peft import PeftModelForCausalLM
from tqdm.auto import tqdm

from data.preprocessor import Preprocessor


@torch.no_grad()
def predict(
        model: PeftModelForCausalLM,
        predict_dataset: Dataset,
        preprocessor: Preprocessor
    ) -> list[dict[str, Any]]:
    pbar = tqdm(total=len(predict_dataset), desc="Eval")
    results = []
    for document in predict_dataset:
        pbar.update(1)
        for messages in preprocessor.get_messages(document):
            model_input = messages[:-1], messages[-1]
            gold_output = messages[-1]['content']
            tokenized_chat = preprocessor.tokenizer.apply_chat_template(
                model_input,
                tokenize = True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            tokenized_chat = tokenized_chat.to(model.device)
            generated_tokens = model.generate(tokenized_chat, max_new_tokens=512)
            generated_text = preprocessor.tokenizer.decode(generated_tokens[0]).replace(preprocessor.tokenizer.eos_token, "\n")
            generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            golds = preprocessor.parse_output(gold_output)
            preds = preprocessor.parse_output(generated_text)
            results.append({"id": document["id"], "text": model_input[1]["content"], "golds": golds, "preds": preds, 'generated_text': generated_text})

    return results


def submit_wandb_predict(predictions: list[dict[str, Any]]) -> None:
    columns = ["id", "text", "gold", "predictions", "generated_text"]
    result_table = wandb.Table(columns=columns)

    for prediction in predictions:
        result_table.add_data(
            prediction["id"],
            prediction["text"],
            ', '.join(prediction["golds"]),
            ', '.join(prediction["preds"]),
            prediction["generated_text"]
        )
    wandb.log({"predictions": result_table})
