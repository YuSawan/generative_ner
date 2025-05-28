
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from src.data import Preprocessor

dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = load_dataset("json", data_files={"train": dataset_path}, cache_dir='tmp/')
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels2names = {label: label for label in sorted(label_set)}

MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-3-1b-it", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen3-0.6B"]
FORMATS = ["collective", "individual", "universal"]


class TestPreprocessor:
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("format", FORMATS)
    @pytest.mark.parametrize("language", ["en", "ja"])
    def test___call__(self, model: str, format: str, language: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        preprocessor = Preprocessor(
            tokenizer = tokenizer,
            labels2names=labels2names,
            language = language,
            format = format
        )

        if model == "google/gemma-3-1b-it":
            assert preprocessor.response_template == '<start_of_turn>model\n'
        if model == "mistralai/Mistral-7B-v0.1":
            assert preprocessor.response_template == '[/INST]'
        if model == "Qwen/Qwen3-0.6B":
            assert preprocessor.response_template == '<|im_start|>assistant\n'
        if model == "meta-llama/Meta-Llama-3-8B-Instruct":
            assert preprocessor.response_template == '<|start_header_id|>assistant<|end_header_id|>'

        for document in raw_datasets['train']:
            preprocessor(document)

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("format", FORMATS)
    @pytest.mark.parametrize("language", ["en", "ja"])
    def test_segment(self, model: str, format: str, language: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        preprocessor = Preprocessor(
            tokenizer = tokenizer,
            labels2names=labels2names,
            language = language,
            format = format
        )

        cnt = 0
        for document in raw_datasets['train']:
            for _ in preprocessor.segment(document["examples"]):
                cnt += 1
        assert cnt == 8

