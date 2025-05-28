import json
import re
import string
from typing import Any, Iterator, Optional, TypedDict

from datasets import Dataset, DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizer, TrainingArguments

CHAT_TEMPLATE = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% set content = bos_token + message['role'] + ': ' + message['content'] + eos_token %}
{{ content }}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token + 'assistant: ' }}
{% endif %}\
"""


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Example(TypedDict):
    id: str
    text: str
    entities: list[Entity]
    word_positions: Optional[list[tuple[int, int]]]


def normalize_answer(s: str) -> str:
    ## Universal NER
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation) - set([":"])
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def single_parser(text: str) -> list[tuple[str] | str]:
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([normalize_answer(element) for element in item])
            else:
                item = normalize_answer(item)
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []


class Preprocessor:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            labels2names: dict[str, str],
            language: str='en',
            format: str = 'collective' # 'individual', or 'universal'
        ) -> None:
        self.tokenizer = tokenizer
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = CHAT_TEMPLATE
            self.response_template = self.tokenizer.bos_token + "assistant: "
        else:
            ## Instruct-models
            self.response_template = "<|start_header_id|>assistant<|end_header_id|>"

        self.labels2names = labels2names
        self.format = format
        self.language = language

    @staticmethod
    def segment(document: list[Example]) -> Iterator[tuple[str, list[tuple[str, str]]]]:
        for example in document:
            entities = [(e['label'], example['text'][e['start']: e['end']]) for e in example['entities']]
            yield example['text'], entities

    def get_messages(self, document: list[Example]) -> Iterator[list[dict[str, str]]]:
        for text, entities in self.segment(document):
            if self.format == 'collective':
                messages = self.get_collective_prompt(text, entities, self.labels2names, self.language)
                yield messages
            elif self.format == 'universal':
                messages = self.get_universal_prompt(text, entities, self.labels2names, self.language)
                yield messages
            elif self.format == 'individual':
                for message in self.get_individual_prompt(text, entities, self.labels2names, self.language):
                    yield message
            else:
                raise NotImplementedError(f"Format '{self.format}' is not implemented.")

    @staticmethod
    def get_collective_prompt(text: str, entities: list[tuple[str, str]], labels2names: dict[str, str], language: str) -> list[dict[str, str]]:
        output = '; '.join([f'{labels2names[label]}: {entext}' for label, entext in entities]) if entities else " None"
        if language == 'ja':
            messages = [
                {"role": "system", "content": 'バーチャルアシスタントは、提供されたテキストに基づいてユーザーの質問に答えます。'},
                {"role": "user", "content": f'テキスト: {text}'},
                {"role": "assistant", "content": 'テキストを読み終えました。'},
                {"role": "user", "content": f'テキストから{'、'.join([label for label in labels2names.values()])}に関連するエンティティをすべて見つけてください。 出力フォーマットは、"type1: word1; type2: word2"です。'},
                {"role": "assistant", "content": output},
            ]
        else:
            messages = [
                {"role": "system", "content": "A virtual assistant answers questions from a user based on the provided text."},
                {"role": "user", "content": f"Text: {text}"},
                {"role": "assistant", "content": 'I’ve read this text.'},
                {"role": "user", "content": f'Please find all the entity words associated with {len(labels2names)} entity types in the given text: {', '.join([label for label in labels2names.values()])}. Output format is "type1: word1; type2: word2".'},
                {"role": "assistant", "content": output},
            ]
        return messages

    @staticmethod
    def get_individual_prompt(text: str, entities: list[tuple[str, str]], labels2names: dict[str, str], language: str) -> Iterator[list[dict[str, str]]]:
        if language == 'ja':
            messages = [
                {"role": "system", "content": "バーチャルアシスタントは、提供されたテキストに基づいてユーザーの質問に答えます。"},
                {"role": "user", "content": f"テキスト: {text}"},
                {"role": "assistant", "content": 'テキストを読み終えました。'},
            ]
        else:
            messages = [
                {"role": "system", "content": "A virtual assistant answers questions from a user based on the provided text."},
                {"role": "user", "content": f"Text: {text}"},
                {"role": "assistant", "content": 'I’ve read this text.'},
            ]

        for label, name in labels2names.items():
            entity_texts = []
            entity_texts = [f'"{entext}"' for enlabel, entext in entities if enlabel == label]
            output = "[" + ', '.join(entity_texts) + "]"
            additional_content = [
                {"role": "user", "content": f"What describes {name} in the text?" if language != 'ja' else f"テキストには何の{name}が述べられているでしょうか？"},
                {"role": "assistant", "content": output},
            ]
            yield messages + additional_content

    @staticmethod
    def get_universal_prompt(text: str, entities: list[tuple[str, str]], labels2names: dict[str, str], language: str) -> list[dict[str, str]]:
        output = '; '.join([f'{labels2names[label]}: {entext}' for label, entext in entities]) if entities else " None"
        if language == 'ja':
            messages = [
                {"role": "system", "content": 'バーチャルアシスタントは、提供されたテキストに基づいてユーザーの質問に答えます。'},
                {"role": "user", "content": f'テキスト: {text}'},
                {"role": "assistant", "content": 'テキストを読み終えました。'},
                {"role": "user", "content": 'テキストから、カテゴリーに関連するエンティティをすべて見つけてください。 出力フォーマットは、"type1: word1; type2: word2"です。'},
                {"role": "assistant", "content": output},
            ]
        else:
            messages = [
                {"role": "system", "content": "A virtual assistant answers questions from a user based on the provided text."},
                {"role": "user", "content": f"Text: {text}"},
                {"role": "assistant", "content": 'I’ve read this text.'},
                {"role": "user", "content": 'Please find all the entity words associated with the category in the given text. Output format is "type1: word1; type2: word2".'},
                {"role": "assistant", "content": output},
            ]
        return messages

    @staticmethod
    def parse_output(output: str, format: str) -> list[str]:
        entities = []
        if format == 'individual':
            for entity in single_parser(output):
                if isinstance(entity, tuple):
                    for e in entity:
                        entities.append(e)
                else:
                    entities.append(entity)
        elif format in ['collective', 'universal']:
            for segment in output.split(";"):
                entities.append(normalize_answer(segment))
        else:
            raise NotImplementedError(f"Format '{format}' is not implemented.")

        return entities

    def __call__(self, document: list[Example]) -> Iterator[str]:
        for messages in self.get_messages(document):
            yield self.tokenizer.apply_chat_template(messages, return_dict='pt')


# class T5Preprocessor(Preprocessor):
#     def get_messages(self, document: list[Example]) -> Iterator[list[dict[str, str]]]:
#         for text, entities in self.segment(document):
#             if self.format == 'single':
#                 messages = self.get_single_prompt(text, entities, self.language)
#                 yield messages
#             else:
#                 for label in self.labels:
#                     messages = self.get_multi_prompt(text, label, entities, self.language)
#                     yield messages

#     def __call__(self, document: list[Example]) -> Iterator[BatchEncoding]:
#         for messages in self.get_messages(document):
#             encoding = self.tokenizer(messages[0], truncation=True, return_tensors='pt')
#             encoding['labels'] = self.tokenizer(messages[1], truncation=True, return_tensors='pt').input_ids
#             yield encoding


#     @staticmethod
#     def get_single_prompt(text: str, entities: list[tuple[str, str]], language: str) -> list[dict[str, str]]:
#         output = '; '.join([f'{label}: {entext}' for label, entext in entities]) if entities else " None"
#         messages = [
#             {"role": "user", "content": f'Please find all the entity words associated with the category in the given text. Output format is "type1: word1; type2: word2": {text}'},
#             {"role": "assistant", "content": output},
#         ]
#         return messages

#     @staticmethod
#     def get_multi_prompt(text: str, labels: str, entities: list[tuple[str, str]], language: str) -> list[dict[str, str]]:
#         entity_texts = []
#         for label, entext in entities:
#             if label == target_label:
#                 entity_texts.append(f'"{entext}"')
#         output = "[" + ', '.join(entity_texts) + "]"
#         messages = [
#             {"role": "user", "content": f'What describes {label} in the text?: {text}'},
#             {"role": "assistant", "content": output},
#         ]
#         return messages


def get_splits(
        raw_datasets: DatasetDict,
        preprocessor: Preprocessor,
        training_arguments: Optional[TrainingArguments]=None
        ) -> dict[str, Dataset]:
    def preprocess(documents: Dataset) -> Any:
        features: list[BatchEncoding] = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    if training_arguments:
        with training_arguments.main_process_first(desc="dataset map pre-processing"):
            column_names = next(iter(raw_datasets.values())).column_names
            splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)
    else:
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)

    return splits
