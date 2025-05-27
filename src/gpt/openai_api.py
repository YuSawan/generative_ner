import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import httpx
import tiktoken
from openai import OpenAI
from openai.types import FileObject
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from .cost import EstimatedCost, count_fee
from .utils import Cache, dump_jsonline, load_jsonline, serialize

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

def check_model_support(model: str) -> None:
    """check_applicable_model

    This function is to check whether the specified model is supported.

    Args:
        results (dict[str, Any]): Generation output of a OpenAI (2024/12/1 version)
        model (str): Model name of OpenAI's GPT models
        use_batchapi (bool): True if you use BatchAPI

    Raises:
        RuntimeError: This occurs when an unsupported GPT model is specified, or when a cheaper version can be used for another model.
    """

    count_fee({'usage': {"prompt_tokens": 50000000, "completion_tokens": 2000000}}, model)


def call_openai_client(api_file: str, timeout: httpx.Timeout | None = None) -> OpenAI:
    api_key = json.load(open(api_file))
    try:
        OPENAI_ORGANIZATION_KEY = api_key["OPENAI_ORGANIZATION_KEY"]
    except KeyError:
        OPENAI_ORGANIZATION_KEY = None
    OPENAI_API_KEY = api_key["OPENAI_API_KEY"]
    client = OpenAI(organization=OPENAI_ORGANIZATION_KEY, api_key=OPENAI_API_KEY, timeout=timeout)
    return client


def convert_messages(messages: list[dict[str, str]]) -> Iterable[ChatCompletionMessageParam]:
    for m in messages:
        if m["role"] == "system":
            system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": m["content"]}
            yield system_message
        elif m["role"] == "user":
            user_message: ChatCompletionUserMessageParam = {"role": "user", "content": m["content"]}
            yield user_message
        elif m["role"] == "assistant":
            assistant_message: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": m["content"]}
            yield assistant_message
        elif m["role"] == "tool":
            tool_message: ChatCompletionToolMessageParam = {"role": "tool", "content": m["content"], "tool_call_id": m["tool_call_id"]}
            yield tool_message
        elif m["role"] == "function":
            function_messages: ChatCompletionFunctionMessageParam = {"role": "function", "content": m["content"], "name": m["name"]}
            yield function_messages
        else:
            raise RuntimeError(f"{m['role']} is not found in the ChatCompletionMessageParam. Correctly assign the role from 'system', 'user', 'assistant', 'tool', and 'function'")


#@retry(wait=wait_exponential(multiplier=1, min=3, max=50))
def generate_by_client(
        client: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.0,
        seed: int = 0,
        n: int = 1,
        max_token_length: int = 100,
        json_object: bool = False
    ) -> ChatCompletion:
    # https://platform.openai.com/docs/api-reference/completions/create

    if json_object:
        response = client.chat.completions.create(
            model=model,
            messages=convert_messages(messages),
            temperature = temperature,
            top_p = top_p,
            seed = seed,
            n = n,
            max_tokens = max_token_length,
            response_format={ "type": "json_object" },
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=convert_messages(messages),
            temperature = temperature,
            top_p = top_p,
            seed = seed,
            n = n,
            max_tokens = max_token_length,
        )
    print(response)

    if response.choices[0].finish_reason != 'stop':
        raise RuntimeError(f"This process was terminated by the OpenAI API due to finish_reason={response.choices[0].finish_reason}")

    return response


def estimate_by_tiktoken(
        model: str,
        messages: list[dict[str, str]],
        gold_output: Optional[str] = None,
    ) -> dict[str, dict[str, int]]:
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = []
    for message in messages:
        prompt_tokens.extend(encoding.encode(message["content"]))

    if gold_output:
        completion_tokens = encoding.encode(gold_output)
    else:
        completion_tokens = []
    return {"usage": {"prompt_tokens": len(prompt_tokens), "completion_tokens": len(completion_tokens)}}


class OpenAI_APIWrapper:
    """OpenAI_APIWrapper

    This wrapper is a inclusive OpenAI API tools.
    I strongly recommend to debug with estimate() at first, then use BatchAPI(ex submit_batch()) in experiment.
    """
    def __init__(
            self,
            model_name: str,
            api_file: str,
            timeout: Optional[httpx.Timeout] = None,
            use_batchapi: Optional[bool] = False
        ) -> None:
        """__init__

        Args:
            model_name (str): OpenAI's model name
            api_file (str): Path of file of OpenAI api token. I assume that the api_token of OpenAI is formatting as json.
            total_cost_limit (float): Limit of total cost. This API terminates the process if the output cost is exceed of the limit. (Currency: USD)
            timeout (Optional[httpx.Timeout]): Timeout setting (Default: None)
            use_batchapi (Optional[bool]): True if use batchapi (Default: False)
        """
        check_model_support(model_name)
        self.client = call_openai_client(api_file, timeout)
        self.model_name = model_name
        self.batch: list[dict[str, Any]] = []
        self.cache = Cache(model_name)
        self.use_batchapi = use_batchapi

    def _generate(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.2,
            max_token_length: int = 4096,
            top_p: float = 0.0,
            seed: int = 0,
            n: int = 1,
            json_format: bool = False,
    ) -> ChatCompletion:
        """_generate

        This function is to call the OpenAI's client.

        Args:
            messages (list[dict[str, str]]): submit body of message, including instruction, demonstration.
            temperature (float): temperature of GPT
            max_token_length: max token length for output
            top_p: probability to choice token
            seed (int): model seed,
            n (int): top_n output,
            json_format (bool): True if you want to generate by json format

        Returns:
            response (ChatCompletion): the called result by client
        Raises:
            RuntimeError: This occurs when the client occurs any of errors.
        Note:
            https://platform.openai.com/docs/api-reference/completions/create
        """

        if json_format:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=convert_messages(messages),
                temperature = temperature,
                top_p = top_p,
                seed = seed,
                n = n,
                max_tokens = max_token_length,
                response_format={ "type": "json_object" },
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=convert_messages(messages),
                temperature = temperature,
                top_p = top_p,
                seed = seed,
                n = n,
                max_tokens = max_token_length,
            )
        if response.choices[0].finish_reason != 'stop':
            raise RuntimeError(f"This process was terminated by the OpenAI API due to finish_reason={response.choices[0].finish_reason}")
        return response

    def generate(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.2,
            max_token_length: int = 4096,
            top_p: float = 0.0,
            seed: int = 0,
            n: int = 1,
            json_format: bool = False,
            debug: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """generate
        This function is to call the OpenAI's client.
        If client has already called the same messages, this function generates the result from .cache/

        Args:
            messages (Iterable[ChatCompletionMessageParam] | list[dict[str, str]]): submit body of message, including instruction, demonstration.
            temperature (float): temperature of GPT
            max_token_length: max token length for output
            top_p: probability to choice token
            seed (int): model seed,
            n (int): top_n output,
            json_format (bool): True if you want to generate by json format
            debug (bool): if True whether debug mode

        Returns:
            result (ChatCompletion): the called result by client
        """

        cached_instruction = '\n'.join([m['content'] for m in messages])
        if not self.cache.check_in_cache(cached_instruction) or debug:
            _results = self._generate(messages, temperature, max_token_length, top_p, seed, n, json_format)
            results = serialize(_results)
            if not debug:
                self.cache.append_to_jsonl(cached_instruction, results)
        else:
            results = self.cache(cached_instruction)
        return results

    def estimate(
            self,
            messages: list[dict[str, str]],
            gold_output: Optional[str] = None,
            model_name: Optional[str] = None
    ) -> dict[str, dict[str, int]]:
        """estimate

            This function is to estimate cost by tiktoken.
            tiktoken is freely get tokens of the prompt, so you can estimate without money.

        Args:
            messages (list[dict[str, str]]): Messages to submit to the OpenAI module
            gold_output (str): Ideal output for messages
            model_name (Optional[str]): Model name of OpenAI
            use_batchapi (bool): True if use BatchAPI (Default: False)

        Returns:
            estimated_tokens (dict[str, dict[str, int]]): estimated number of input/output tokens.
        """

        if not model_name:
            model_name = self.model_name
        return estimate_by_tiktoken(model_name, messages, gold_output)

    def append_messages_to_batch(
            self,
            messages: list[dict[str, str]],
            temperature: float=0.2,
            max_token_length: int = 4096,
            top_p: float = 0.0,
            seed: float = 0,
            n: float = 1,
            json_object: bool=False,
    ) -> None:
        """append_messages_to_batch

        This function is to append messages to a batch for submitting to BatchAPI

        Args:
            messages (list[dict[str, str]]): submit body of message, including instruction, demonstration.
            temperature (float): temperature of GPT
            max_token_length: max token length for output
            top_p: probability to choice token
            seed (int): model seed,
            n (int): top_n output,
            json_format (bool): True if you want to generate by json format
        """
        body = {
            # This is what you would have in your Chat Completions API call
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_token_length,
            "top_p": top_p,
            "seed": seed,
            "n": n,
            "messages": messages,
        }
        if json_object:
            body.update({"response_format": {"type": "json_object"}})

        index = len(self.batch)
        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        self.batch.append(task)

    def estimate_batch(self, max_new_tokens: int = 4096) -> EstimatedCost:
        total_tokens = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        for b in self.batch:
            model_name = b["body"]["model"]
            estimated_tokens = estimate_by_tiktoken(model_name, b["body"]["messages"])
            total_tokens["usage"]["prompt_tokens"] += estimated_tokens["usage"]["prompt_tokens"]
            total_tokens["usage"]["completion_tokens"] += max_new_tokens
        total_cost = count_fee(total_tokens, model_name, use_batchapi=True)
        return total_cost

    def save_batch(self, save_dir: str, file_basename: str = 'batch_input', batch_size: int = 50000, ensure_ascii: bool = True) -> None:
        """save_batch

        This function is to save a batch to a file formatting as jsonline.

        Args:
            save_dir (str): Path to the directory for save batch
            file_name (str): File name of the batch (Default: batch_input.jsonl)
            ensure_ascii (bool): False If you use unicode strings such as Japanese

        """
        os.makedirs(save_dir, exist_ok=True)
        if not self.batch:
            raise RuntimeError("There is no task in self.batch")
        for i, k in enumerate(range(0, len(self.batch), batch_size)):
            dump_jsonline(self.batch[k: k+batch_size], Path(save_dir, f"{file_basename}_{i}.jsonl"), ensure_ascii)


class BatchAPI_Wrapper:
    """BatchAPI_Wrapper

    This wrapper is a inclusive Batch API tools by OpenAI.
    To use this wrapper, in advance, you must save batch data submitting to BatchAPI.
    You can make the batch with OpenAPI_APIWrapper.
    """

    def __init__(
            self,
            api_file: str,
            saved_dir: str,
            file_input_basename: str = "batch_input",
            file_output_basename: str = "batch_output"
    ) -> None:
        """__init__

        Args:
            api_file (str): Path to file of OpenAI api token. I assume that the api_token of OpenAI is formatting as json.
            saved_dir (str): Path to directory of saved by OpenAI_APIWrapper
            file_input_basename (str): Path to saved jsonline file by OpenAI_APIWrapper. (Default: batch_input)
        """
        self.client: OpenAI = call_openai_client(api_file)
        self.saved_dir: str = saved_dir
        self.file_input_basename = file_input_basename
        self.file_output_basename = file_output_basename

    def _upload_batch(self, submit_file: str | os.PathLike) ->  FileObject:
        """_upload_batch

        This function is to uoload a jsonline file consisting a batch.

        Args:
            submit_file (str | os.PathLike): Path to the submit file

        Return:
            batch_file (FileObject): uploaded batch file
        """

        batch_file = self.client.files.create(
            file=open(Path(submit_file), "rb"),
            purpose="batch"
        )
        return batch_file

    def submit_batch(self, description: str = "nightly eval job") -> None:
        """submit_batch

        This function is to submit a batch file to BatchAPI

        Args:
            batch_file (FileObject): batch file for submit to BatchAPI
            description (str) : description of batch job (Default: nightly eval job)

        Return:
            batch_file (Batch): loaded batch file
        """

        files = glob.glob(f"{self.saved_dir}/{self.file_input_basename}_*.jsonl")
        if not files:
            raise FileNotFoundError

        batch_jobs_info = []
        for fname in files:
            batch_file = self._upload_batch(fname)
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": description
                }
            )
            input_file_name = self.client.files.retrieve(batch_job.input_file_id).filename
            batch_jobs_info.append({"batch_job_id": batch_job.id, "input_file_name": input_file_name})
            logger.info(f"Submit Job to API. Job ID: {batch_job.id}")

        dump_jsonline(batch_jobs_info, Path(self.saved_dir, 'batch_id_list.jsonl'))

    def get_batch_id(self) -> list[dict[str, str]]:
        batch_list_path = Path(self.saved_dir, 'batch_id_list.jsonl')
        if not Path.exists(batch_list_path):
            raise FileNotFoundError

        return [json.loads(line) for line in open(batch_list_path)]

    def check_job_status(self, batch_job_id: str) -> Literal['validating', 'failed', 'in_progress', 'finalizing', 'completed', 'expired', 'cancelling', 'cancelled']:
        """check_job_status

        This function is to check a batch job status.

        Args:
            batch_file (Batch): submitted batch file
        """
        batch_job = self.client.batches.retrieve(batch_job_id)
        return batch_job.status

    def retrieve_job(self, batch_job_id: str) -> list[dict[str, Any]]:
        """retrieve_job

        This function is to retrieve a result of BatchAPI.

        Args:
            batch_job_id (str): job_id(batch_job.id) of batch job to retrieve

        Returns:
            return (_legacy_response.HttpxBinaryResponseContent): retrieved results of BatchAPI from job_id
        """
        batch_job = self.client.batches.retrieve(batch_job_id)
        result_file_id = batch_job.output_file_id
        if result_file_id:
            result = self.client.files.content(result_file_id)
        else:
            raise RuntimeError("batch_job.output_file_id is not found. Please check whether the batch process is truly finished.")

        input_file_name = self.client.files.retrieve(batch_job.input_file_id).filename
        output_file_name = Path(self.saved_dir, input_file_name.replace(self.file_input_basename, self.file_output_basename))
        result.write_to_file(output_file_name)
        _results = load_jsonline(output_file_name)
        results = [serialize(_result) for _result in _results]

        return results


    def load_batch(self, batch_data: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        input_filename = batch_data['input_file_name']
        model_inputs = load_jsonline(Path(self.saved_dir, input_filename))

        result_filepath = input_filename.replace(self.file_input_basename, self.file_output_basename)
        if Path(self.saved_dir, result_filepath).exists():
            predictions = load_jsonline(Path(self.saved_dir, result_filepath))
        else:
            predictions = self.retrieve_job(batch_data['batch_job_id'])

        return model_inputs, predictions
