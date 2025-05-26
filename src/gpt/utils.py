import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(150.0, read=150.0, write=50.0, connect=6.0)


def dump_jsonline(data: list[dict[str, Any]] | bytes, save_dir: str, file_output_name: str, ensure_ascii: bool = True) -> None:
    """dump_jsonline

    This function is to save data to a file formatting as jsonline.

    Args:
        data (list[dict[str, Any] | bytes]): data for dumping
        save_dir (str): Path to the directory for saving
        file_name (str): File name for saving
        ensure_ascii (bool): False If you use unicode strings such as Japanese
    """
    if isinstance(data, bytes):
        with open(Path(save_dir, file_output_name), 'wb') as file:
            file.write(data)
    else:
        with open(Path(save_dir, file_output_name), 'w') as file:
            for obj in data:
                file.write(json.dumps(obj, ensure_ascii=ensure_ascii) + '\n')


def load_jsonline(saved_dir: str, file_input_name: str) -> list[dict[str, Any]]:
    """load_json

    This function is to simply load a jsonline.

    Args:
        saved_dir (str): Path to the target directory
        file_name (str): File name in the target directory

    Return:
        data (list[dict[str, Any]]): loaded data
    """
    batch = []
    with open(Path(saved_dir, file_input_name), 'r') as file:
        for line in file:
            # line = line.strip().encode('latin1').decode('utf-8')
            json_object = json.loads(line.strip())
            batch.append(json_object)
    return batch


def serialize(obj: Any) -> Any:
    """ 再帰的にオブジェクトを辞書にシリアライズする """
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return serialize(obj.__dict__)
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    else:
        return obj


def create_hash(prompt: str) -> str:
    # instruction単位で保存
    return hashlib.md5(prompt.encode()).hexdigest()


class Cache:
    def __init__(self, model_name: str) -> None:
        # GPTのoutputを保存
        os.makedirs('.cache', exist_ok = True)
        self.file_name_cache = f'.cache/{model_name}'
        if os.path.exists(self.file_name_cache):
            # ここにはハッシュ値と生データが格納される
            self.cache = self.load_json()
        else:
            self.cache = dict()

    def load_json(self) -> dict[str, Any]:
        data = dict()
        with open(self.file_name_cache) as f:
            json_lines = f.readlines()
            for line in json_lines:
                json_obj = json.loads(line)
                data[json_obj['id']] = json_obj['results']
        return data

    def append_to_jsonl(self, instruction: str, results: dict[str, Any]) -> None:
        # データをcacheとして保存。
        self.cache[create_hash(instruction)] = results
        save_data = {'id': create_hash(instruction), 'results': results}
        with open(self.file_name_cache, 'a') as file:
            json_str = json.dumps(save_data, ensure_ascii=False)
            file.write(json_str + '\n')

    def __call__(self, item: str) -> dict[str, Any]:
        return self.cache[create_hash(item)]

    def check_in_cache(self, instruction: str) -> bool:
        return True if create_hash(instruction) in self.cache else False


# def check_llama_tokenize(collator: DataCollatorForCompletionOnlyLM, dataset: Dataset, tokenizer: LlamaTokenizerFast) -> None:
#     import itertools
#     batch = collator(dataset[:1])
#     input_ids = batch['input_ids'][0]
#     labels = batch['labels'][0]
#     segments_to_fit = []
#     segments_to_ignore = []
#     for key, group in itertools.groupby(range(len(input_ids)), key=lambda i: labels[i] == -100):
#         group_list = list(group)
#         if key:
#             segments_to_ignore.append(group_list)
#         else:
#             segments_to_fit.append(group_list)
#     for seg in segments_to_ignore:
#         print(tokenizer.decode(input_ids[seg]))
#         print()
#     for seg in segments_to_fit:
#         print(tokenizer.decode(input_ids[seg]))
#         print()
