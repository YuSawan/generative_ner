import json
import re
import string
from typing import Iterable, Optional, TypedDict

from transformers import BatchEncoding, PreTrainedTokenizer

CHAT_TEMPLATE = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% set content = bos_token + message['role'] + ': ' message['content'] + eos_token %}
{% if loop.index0 == 0 %}
{% set content = bos_token + content + eos_token %}
{% endif %}
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



def _format(s: str) -> str:
    ## Instruct UIE
    s = ' '.join(s.split())
    s = re.sub(r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r'\1', s)
    s = s.lower()
    s = s.replace('{','').replace('}','')
    s = re.sub(',+', ',', s)
    s = re.sub('\.+', '.', s)
    s = re.sub(';+', ';', s)
    s = s.replace('’', "'")
    s = s.replace('location', 'located')
    return s


def normalize_answer(s: str) -> str:
    ## Universal NER
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
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
            labels: list[str],
            language: str='en',
            extend_context: Optional[bool] = False,
            format: str = 'single' # 'single' or 'multi'
        ) -> None:
        self.tokenizer = tokenizer
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = CHAT_TEMPLATE
        self.labels = labels
        self.extend_context = extend_context
        self.format = format # 'single' turn or 'multi' turn
        self.language = language

    def segment(self, document: list[Example]) -> Iterable[tuple[str, list[tuple[str, str]]]]:
        text, entities = "", []
        for example in document:
            entities = [(e['label'], example['text'][e['start']: e['end']]) for e in example['entities']]
            if not self.extend_context:
                yield example['text'], entities
            text += example['text']
            entities.extend(entities)
        if self.extend_context:
            yield text, entities

    def get_messages(self, document: list[Example]) -> Iterable[list[dict[str, str]]]:
        for text, entities in self.segment(document):
            if self.format == 'single':
                messages = self.get_single_prompt(text, entities, self.language)
                yield messages
            else:
                for label in self.labels:
                    messages = self.get_multi_prompt(text, label, entities, self.language)
                    yield messages

    @staticmethod
    def get_single_prompt(text: str, entities: list[tuple[str, str]], language: str) -> list[dict[str, str]]:
        output = "[" + '; '.join([f'{label}: {text}' for text, label in entities]) + "]"
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
    def get_multi_prompt(text: str, target_label: str, entities: list[tuple[str, str]], language: str) -> list[dict[str, str]]:
        entity_texts = []
        for text, label in entities:
            if label == target_label:
                entity_texts.append(f'"{text}"')
        output = "[" + ', '.join(entity_texts) + "]"

        if language == 'ja':
            messages = [
                {"role": "システム", "content": "バーチャルアシスタントは、提供されたテキストに基づいてユーザーの質問に答えます。"},
                {"role": "ユーザ", "content": f"テキスト: {text}"},
                {"role": "アシスタント", "content": 'テキストを読み終えました。'},
                {"role": "ユーザ", "content": f"テキストには何の{label}が述べられているでしょうか？"},
                {"role": "アシスタント", "content": output},
            ]
        else:
            messages = [
                {"role": "system", "content": "A virtual assistant answers questions from a user based on the provided text."},
                {"role": "user", "content": f"Text: {text}"},
                {"role": "assistant", "content": 'I’ve read this text.'},
                {"role": "user", "content": f"What describes {label} in the text?"},
                {"role": "assistant", "content": output},
            ]
        return messages

    def parse_output(self, output: str) -> list[str]:
        entities = []
        if self.format == 'single':
            for segment in output.split(";"):
                entities.append(_format(segment))
        else:
            for entity in single_parser(output):
                if isinstance(entity, tuple):
                    for e in entity:
                        entities.append(e)
                else:
                    entities.append(entity)
        return entities


    def __call__(self, document: list[Example]) -> Iterable[BatchEncoding]:
        for messages in self.get_messages(document):
            yield self.tokenizer.apply_chat_template(messages)



class T5Preprocessor(Preprocessor):
    def get_messages(self, document: list[Example]) -> Iterable[list[dict[str, str]]]:
        for text, entities in self.segment(document):
            if self.format == 'single':
                messages = self.get_single_prompt(text, entities, self.language)
                yield messages
            else:
                for label in self.labels:
                    messages = self.get_multi_prompt(text, label, entities, self.language)
                    yield messages

    def __call__(self, document: list[Example]) -> Iterable[BatchEncoding]:
        for messages in self.get_messages(document):
            encoding = self.tokenizer(messages[0], truncation=True, return_tensors='pt')
            encoding['labels'] = self.tokenizer(messages[1], truncation=True, return_tensors='pt').input_ids
            yield encoding


    @staticmethod
    def get_single_prompt(text: str, entities: list[tuple[str, str]], language: str) -> list[dict[str, str]]:
        output = "[" + '; '.join([f'{label}: {text}' for text, label in entities]) + "]"
        messages = [
            {"role": "user", "content": f'Please find all the entity words associated with the category in the given text. Output format is "type1: word1; type2: word2": {text}'},
            {"role": "assistant", "content": output},
        ]
        return messages

    @staticmethod
    def get_multi_prompt(text: str, target_label: str, entities: list[tuple[str, str]], language: str) -> list[dict[str, str]]:
        entity_texts = []
        for text, label in entities:
            if label == target_label:
                entity_texts.append(f'"{text}"')
        output = "[" + ', '.join(entity_texts) + "]"
        messages = [
            {"role": "user", "content": f'What describes {label} in the text?: {text}'},
            {"role": "assistant", "content": output},
        ]
        return messages
