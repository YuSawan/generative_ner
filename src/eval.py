import torch
from datasets import Dataset
from peft import PeftModelForCausalLM
from tqdm.auto import tqdm

from preprocessor import Preprocessor


@torch.no_grad()
def evaluate(
        model: PeftModelForCausalLM,
        eval_dataset: Dataset,
        preprocessor: Preprocessor
    ) -> dict[str, float]:
    pbar = tqdm(total=len(eval_dataset), desc="Eval")
    n_correct, n_pred, n_gold = 0, 0, 0
    for document in eval_dataset:
        pbar.update(1)
        for messages in preprocessor.get_messages(document['examples']):
            model_input = messages[:-1]
            gold_output = messages[-1]['content']
            prompt = preprocessor.tokenizer.apply_chat_template(model_input, tokenize = False, add_generation_prompt=True)
            tokenized_chat = preprocessor.tokenizer(prompt, return_tensors='pt', padding=True)
            tokenized_chat = tokenized_chat.to(model.device)
            generated_tokens = model.generate(**tokenized_chat, max_new_tokens=512, pad_token_id=preprocessor.tokenizer.eos_token_id)
            generated_text = preprocessor.tokenizer.decode(generated_tokens[0]).replace(preprocessor.tokenizer.eos_token, "\n")
            generated_text = generated_text.split(preprocessor.response_template)[-1].strip()
            golds = preprocessor.parse_output(gold_output)
            preds = preprocessor.parse_output(generated_text)
            n_gold += len(set(golds))
            n_pred += len(set(preds))
            n_correct += len(set(preds).intersection(set(golds)))
    prec = n_correct / (n_pred + 1e-10)
    recall = n_correct / (n_gold + 1e-10)
    f1 = 2 * prec * recall / (prec + recall + 1e-10)
    return {
        'precision': prec,
        'recall': recall,
        'f1': f1,
    }
