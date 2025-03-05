
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from src.preprocessor import Preprocessor

dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = load_dataset("json", data_files={"train": dataset_path}, cache_dir='tmp/')
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])


class TestPreprocessor:
    @pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B-Instruct", "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"])
    @pytest.mark.parametrize("format", ["single", "multi"])
    @pytest.mark.parametrize("language", ["en", "ja"])
    def test___call__(self, model: str, format: str, language: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        preprocessor = Preprocessor(
            tokenizer = tokenizer,
            labels = sorted(label_set),
            language = language,
            format = format
        )

        for document in raw_datasets['train']:
            preprocessor(document)

    @pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B-Instruct", "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"])
    @pytest.mark.parametrize("extend_context", [True, False])
    def test_segment(self, model: str, extend_context: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        preprocessor = Preprocessor(
            tokenizer = tokenizer,
            labels = sorted(label_set),
            extend_context=extend_context,
        )

        cnt = 0
        for document in raw_datasets['train']:
            for _ in preprocessor.segment(document):
                cnt += 1
        if extend_context:
            assert cnt == 5
        else:
            assert cnt == 8

