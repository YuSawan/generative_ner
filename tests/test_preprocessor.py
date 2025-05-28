
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
labels = sorted(label_set)

MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct", "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1", "google/gemma-3-1b-it"]
FORMATS = ["single", "multi", "inclusive"]


class TestPreprocessor:
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("format", FORMATS)
    @pytest.mark.parametrize("language", ["en", "ja"])
    def test___call__(self, model: str, format: str, language: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        preprocessor = Preprocessor(
            tokenizer = tokenizer,
            labels = labels,
            language = language,
            format = format
        )

        for document in raw_datasets['train']:
            preprocessor(document)

    @pytest.mark.parametrize("model", MODELS)
    def test_segment(self, model: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        preprocessor = Preprocessor(
            tokenizer = tokenizer,
            labels = labels,
        )

        cnt = 0
        for document in raw_datasets['train']:
            for _ in preprocessor.segment(document):
                cnt += 1
        assert cnt == 8

