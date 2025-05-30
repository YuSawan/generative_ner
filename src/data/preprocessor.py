import ast
import random
import re
import string
from typing import Any, Iterator, Optional, TypedDict

from datasets import Dataset, DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizer, TrainingArguments


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


def parser(text: str) -> list[tuple[str, str] | str]:
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = ast.literal_eval(text)
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
            language: str='en', # 'ja' for Japanese, 'en' for English
            format: str = 'collective', # 'individual', or 'universal',
            system_message: Optional[str] = None,
        ) -> None:
        assert tokenizer.chat_template
        self.tokenizer = tokenizer
        self.labels2names = labels2names
        self.format = format
        self.language = language
        self.system_message = system_message

        self.instruction_template = None
        if format == 'individual':
            if '[INST]' in tokenizer.chat_template and '[/INST]' in tokenizer.chat_template:
                # mistral series
                self.instruction_template = '[INST]'
            elif '<|start_header_id|>' in tokenizer.chat_template and '<|end_header_id|>' in tokenizer.chat_template:
                # llama series
                self.instruction_template = '<|start_header_id|>user<|end_header_id|>'
            elif '<start_of_turn>' in tokenizer.chat_template and '<end_of_turn>' in tokenizer.chat_template:
                # gemma series
                self.instruction_template = '<start_of_turn>user\n'
            elif '<|im_start|>' in tokenizer.chat_template and '<|im_end|>' in tokenizer.chat_template:
                # chatml format (e.g., Qwen series)
                self.instruction_template = '<|im_start|>user\n'
            else:
                raise NotImplementedError(f"Unknown chat template format: {tokenizer.chat_template}.")

        if '[INST]' in tokenizer.chat_template and '[/INST]' in tokenizer.chat_template:
            # mistral series
            self.response_template = '[/INST]'
        elif '<|start_header_id|>' in tokenizer.chat_template and '<|end_header_id|>' in tokenizer.chat_template:
            # llama series
            self.response_template = '<|start_header_id|>assistant<|end_header_id|>'
        elif '<start_of_turn>' in tokenizer.chat_template and '<end_of_turn>' in tokenizer.chat_template:
            # gemma series
            self.response_template = '<start_of_turn>model\n'
        elif '<|im_start|>' in tokenizer.chat_template and '<|im_end|>' in tokenizer.chat_template:
            # chatml format (e.g., Qwen series)
            self.response_template = '<|im_start|>assistant\n'
        else:
            raise NotImplementedError(f"Unknown chat template format: {tokenizer.chat_template}.")

    def get_messages(self, text: str, entities: list[Entity], shuffle: bool = False) -> list[dict[str, str]]:
        if self.format == 'collective':
            return self.get_collective_prompt(text, entities, self.labels2names, self.language, self.system_message)
        elif self.format == 'universal':
            return self.get_universal_prompt(text, entities, self.labels2names, self.language, self.system_message)
        elif self.format == 'individual':
            return self.get_individual_prompt(text, entities, self.labels2names, self.language, self.system_message, shuffle)
        else:
            raise NotImplementedError(f"Format '{self.format}' is not implemented.")

    @staticmethod
    def get_collective_prompt(text: str, entities: list[Entity], labels2names: dict[str, str], language: str, system_message: Optional[str] = None) -> list[dict[str, str]]:
        entity_list = list(set([(text[e["start"]: e["end"]], labels2names[e["label"]]) for e in entities]))
        output = "[" + ', '.join([f'("{mention}", "{label}")' for mention, label in entity_list]) + "]"
        messages = [{"role": "system", "content": system_message}] if system_message else []
        if language == 'ja':
            messages.extend([
                {"role": "user", "content": f'テキストからカテゴリに関連するすべてのエンティティを見つけてください。 出力は以下の形式のタプルのリストにしてください： [("entity 1", "type of entity 1"), ... ]。\nOption: {', '.join([label for label in labels2names.values()])}。\nText: {text}'},
                {"role": "assistant", "content": output},
            ])
        elif language == 'en':
            messages.extend([
                {"role": "user", "content": f'Find all the entities associated with the category in the text. The output should be in a list of tuples of the following format: [("entity 1", "type of entity 1"), ... ]\nOption: {', '.join([label for label in labels2names.values()])}.\nText: {text}'},
                {"role": "assistant", "content": output},
            ])
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages are 'ja' and 'en'.")

        return messages

    @staticmethod
    def get_universal_prompt(text: str, entities: list[Entity], labels2names: dict[str, str], language: str, system_message: Optional[str] = None) -> list[dict[str, str]]:
        entity_list = list(set([(text[e["start"]: e["end"]], labels2names[e["label"]]) for e in entities]))
        output = "[" + ', '.join([f'("{mention}", "{label}")' for mention, label in entity_list]) + "]"
        messages = [{"role": "system", "content": system_message}] if system_message else []
        if language == 'ja':
            messages.extend([
                {"role": "user", "content": f'与えられたテキストからすべてのエンティティを抽出し、エンティティタイプを識別してください。 出力は以下の形式のタプルのリストにしてください： [("entity 1", "type of entity 1"), ... ]。\nテキスト: {text}'},
                {"role": "assistant", "content": output},
            ])
        elif language == 'en':
            messages.extend([
                {"role": "user", "content": f'Given a passage, your task is to extract all entities and identify their entity types from the text. The output should be in a list of tuples of the following format: [("entity 1", "type of entity 1"), ... ]\nPassage: {text}'},
                {"role": "assistant", "content": output},
            ])
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages are 'ja' and 'en'.")
        return messages

    @staticmethod
    def get_individual_prompt(text: str, entities: list[Entity], labels2names: dict[str, str], language: str, system_message: Optional[str] = None, shuffle: Optional[bool] = False) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_message}] if system_message else []
        if language == 'ja':
            messages.extend([
                {"role": "user", "content": f'テキスト: {text}'},
                {"role": "assistant", "content": 'テキストを読み終えました。'},
            ])
        elif language == 'en':
            messages.extend([
                {"role": "user", "content": f"Text: {text}"},
                {"role": "assistant", "content": 'I’ve read this text.'},
            ])
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages are 'ja' and 'en'.")

        labels = list(labels2names.keys())
        if shuffle:
            random.shuffle(labels)

        for label in labels:
            name = labels2names[label]
            entity_list = list(set([(text[e["start"]: e["end"]], labels2names[e["label"]]) for e in entities if e["label"] == label]))
            output = "[" + ', '.join([f'"{mention}"' for mention, _ in entity_list]) + "]"
            if language == 'ja':
                messages.extend([
                    {"role": "user", "content": f"テキストには何の{name}が述べられていますか？"},
                    {"role": "assistant", "content": output},
                ])
            elif language == 'en':
                messages.extend([
                    {"role": "user", "content": f"What describes {name} in the text?"},
                    {"role": "assistant", "content": output},
                ])
            else:
                raise ValueError(f"Unsupported language: {language}. Supported languages are 'ja' and 'en'.")
        return messages

    @staticmethod
    def parse_output(output: str) -> list[tuple[str, str] | str]:
        entities = []
        for line in output.split('\n'):
            for entity in parser(line):
                entities.append(entity)
        return entities

    def __call__(self, document: list[Example]) -> Iterator[str]:
        for example in document:
            messages = self.get_messages(example["text"], example["entities"], shuffle=True)
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
