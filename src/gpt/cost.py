import dataclasses
from typing import Any, Optional

fx_rate = 150.0

@dataclasses.dataclass
class EstimatedCost:
    usd: float
    jpy: float
    prompt_tokens: int
    completion_tokens: int


class CostError(Exception):
    pass


def cost_calculate_gpt4o_mini(results: dict[str, dict[str, int]], model: str) -> float:
    """cost_calculate_gpt4o_mini

    This module is to estimate cost of output gpt-4o-mini models (2024/9/26 version)

    Args:
        results (dict[str, dict[str, int]]): Generation output of a OpenAI (2024/9/26 version)
        model (str): Model name of gpt-4o-mini models

    Returns:
        fee (fee): Result of OpenAI models. The result is total fee and input/output tokens.\n

    Raises:
        RuntimeError: This occurs when an unsupported GPT model is specified, or when a cheaper version can be used for another model.
    Note:
        This calculation module summarizes the cost calculation of the GPT model as of 2024/9/26.
        GPT models after 2024/9/26 are not supported, so please update as needed.
    """
    if model in ['gpt-4o-mini', 'gpt-4o-mini-2024-07-18']:
        fee = 0.00015 * results['usage']["prompt_tokens"] / 1000 + 0.0006 * results['usage']["completion_tokens"] / 1000
    else:
        raise RuntimeError("Your specified model is unknown. Search the price of the model and add its pricing to this module!!!")
    return fee


def cost_calculate_gpt4o(results: dict[str, dict[str, int]], model: str) -> float:
    """cost_calculate_gpt4o

    This module is to estimate cost of output gpt-4o models (2024/9/26 version)

    Args:
        results (dict[str, dict[str, int]]): Generation output of a OpenAI (2024/9/26 version)
        model (str): Model name of gpt-4o models

    Returns:
        fee (fee): Result of OpenAI models. The result is total fee and input/output tokens.\n

    Raises:
        RuntimeError: This occurs when an unsupported GPT model is specified, or when a cheaper version can be used for another model.
    Note:
        This calculation module summarizes the cost calculation of the GPT model as of 2024/9/26.
        GPT models after 2024/9/26 are not supported, so please update as needed.
    """
    if model in ['gpt-4o', 'gpt-4o-2024-05-13']:
        fee = 0.005 * results['usage']["prompt_tokens"] / 1000 + 0.015 * results['usage']["completion_tokens"] / 1000
        best_fee = 0.0025 * results['usage']["prompt_tokens"] / 1000 + 0.010 * results['usage']["completion_tokens"] / 1000
        raise RuntimeError(f"There is a cheaper model than your specified (Your specified: {fee}USD, Best: {best_fee}USD). I recommend to use 'gpt-4o-2024-08-06'.")
    elif model in ['gpt-4o-2024-08-06', 'gpt-4o-2024-11-20']:
        fee = 0.0025 * results['usage']["prompt_tokens"] / 1000 + 0.010 * results['usage']["completion_tokens"] / 1000
    else:
        raise RuntimeError("Your specified model is unknown. Search the price of the model and add its pricing to this module!!!")
    return fee


def cost_calculate_o1_mini(results: dict[str, dict[str, int]], model: str) -> float:
    """cost_calculate_gpt4o

    This module is to estimate cost of output gpt-4o models (2024/9/26 version)

    Args:
        results (dict[str, dict[str, int]]): Generation output of a OpenAI (2024/9/26 version)
        model (str): Model name of gpt-4o models

    Returns:
        fee (fee): Result of OpenAI models. The result is total fee and input/output tokens.\n

    Raises:
        RuntimeError: This occurs when an unsupported GPT model is specified, or when a cheaper version can be used for another model.
    Note:
        This calculation module summarizes the cost calculation of the GPT model as of 2024/9/26.
        GPT models after 2024/9/26 are not supported, so please update as needed.
    """
    if model in ['o1-mini', 'o1-mini-2024-09-12']:
        fee = 0.003 * results['usage']["prompt_tokens"] / 1000 + 0.012 * results['usage']["completion_tokens"] / 1000
    else:
        raise RuntimeError("Your specified model is unknown. Search the price of the model and add its pricing to this module!!!")
    return fee


def cost_calculate_gpt3_5_turbo(results: dict[str, dict[str, int]], model: str) -> float:
    """cost_calculate_gpt3_5

    This module is to estimate cost of output gpt-3.5-turbo models (2024/9/26 version)

    Args:
        results (dict[str, dict[str, int]]): Generation output of a OpenAI (2024/9/26 version)
        model (str): Model name of gpt-3.5-turbo models

    Returns:
        fee (fee): Result of OpenAI models. The result is total fee and input/output tokens.\n

    Raises:
        RuntimeError: This occurs when an unsupported GPT model is specified, or when a cheaper version can be used for another model.
    Note:
        This calculation module summarizes the cost calculation of the GPT model as of 2024/9/26.
        GPT models after 2024/9/26 are not supported, so please update as needed.
    """
    if model in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301"]:
        best_fee = 0.0005 * results['usage']["prompt_tokens"] / 1000 + 0.0015 * results['usage']["completion_tokens"] / 1000
        if model in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301"]:
            fee = 0.0015 * results['usage']["prompt_tokens"] / 1000 + 0.002 * results['usage']["completion_tokens"] / 1000
        elif model in ["gpt-3.5-turbo-1106"]:
            fee = 0.001 * results['usage']["prompt_tokens"] / 1000 + 0.002 * results['usage']["completion_tokens"] / 1000
        elif model in ["gpt-3.5-turbo-16k-0613"]:
            fee = 0.003 * results['usage']["prompt_tokens"] / 1000 + 0.004 * results['usage']["completion_tokens"] / 1000
        raise RuntimeError(f"There is a cheaper model than your specified (Your specified: {fee}USD, Best: {best_fee}USD). I recommend to use 'gpt-3.5-turbo-0125'.")
    elif model == "gpt-3.5-turbo-0125":
        fee = 0.0005 * results['usage']["prompt_tokens"] / 1000 + 0.0015 * results['usage']["completion_tokens"] / 1000
    else:
        raise RuntimeError("Your specified model is unknown. Search the price of the model and add its pricing to this module!!!")
    return fee


def count_fee(results: dict[str, Any], model: str, use_batchapi: Optional[bool] = False) -> EstimatedCost:
    """count_fee

    This module is to estimate cost of output OpenAI models (2024/12/1 version)

    Args:
        results (dict[str, Any]): Generation output of a OpenAI (2024/12/1 version)
        model (str): Model name of OpenAI's GPT models

    Returns:
        fee_dict (EstimatedCost): Result of OpenAI models. The result is total fee and input/output tokens.\n
        keys:
        "usd" (float): Estimated Costs(USD$)
        "jpy" (float): Estimated Costs(JPY¥)
        "prompt_tokens" (int): Number of prompt tokens
        "completion_tokens" (int): Number of generated tokens

    Raises:
        RuntimeError: This occurs when an unsupported GPT model is specified, or when a cheaper version can be used for another model.
    Note:
        This calculation module summarizes the cost calculation of the GPT model as of 2024/12/1.
        GPT models after 2024/12/1 are not supported, so please update as needed.
    """
    if model.startswith('gpt-4o-mini'):
        fee = cost_calculate_gpt4o_mini(results, model)
    elif model.startswith('gpt-4o'):
        fee = cost_calculate_gpt4o(results, model)
    elif model.startswith('gpt-3.5-turbo'):
        fee = cost_calculate_gpt3_5_turbo(results, model)
    # elif model.startswith('o1-mini'):
    #     fee = cost_calculate_o1_mini(results, model)
    else:
        raise RuntimeError("Your specified model is unknown. Search the price of the model and add its pricing to this module!!!")

    if use_batchapi:
        fee_dict = EstimatedCost(usd=fee/2, jpy=fee*fx_rate/2, prompt_tokens=results['usage']["prompt_tokens"], completion_tokens=results['usage']["completion_tokens"])
    else:
        fee_dict = EstimatedCost(usd=fee, jpy=fee*fx_rate, prompt_tokens=results['usage']["prompt_tokens"], completion_tokens=results['usage']["completion_tokens"])

    return fee_dict


class CostChecker:
    """CostEstimator

    This wrapper is to estimate the total of cost with OpenAI models.
    """

    def __init__(self, model_name: str, cost_usd_limit: float, use_batchapi: bool = False, estimate: bool = False):
        """__init__

        Args:
            model_name (str): OpenAI's model name
            cost_usd_limit (float): Limit of total cost of USD. This process terminates the process if the total cost of outputs is exceed of the limit.
            use_batchapi (bool): True if use batchapi (Default: False)
            estimate (bool): True if use estimate mode (Default: False)
        """
        self.model_name = model_name
        self.usd_limit = cost_usd_limit
        self.use_batchapi = use_batchapi
        self.estimate = estimate
        self.total_usd = 0.
        self.total_jpy = 0.
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, results: dict[str, Any]) -> EstimatedCost:
        """__call__

        This function is to calculate costs of the results.

        Args:
            results (dict[str, Any]): Generation output of a OpenAI

        Returns:
            fee_dict (EstimatedCost): Result of OpenAI models. The result is total fee and input/output tokens.\n
            keys:
                "usd" (float): Estimated Costs(USD$)
                "jpy" (float): Estimated Costs(JPY¥)
                "prompt_tokens" (int): Number of prompt tokens
                "completion_tokens" (int): Number of generated tokens
        """
        fee_dict = count_fee(results, self.model_name, self.use_batchapi)
        self.total_usd += fee_dict.usd
        self.total_jpy += fee_dict.jpy
        self.prompt_tokens += fee_dict.prompt_tokens
        self.completion_tokens += fee_dict.completion_tokens

        if not self.estimate:
            if self.total_usd >= self.usd_limit:
                raise CostError("Total USD cost have exceeded your budget.")

        return fee_dict

    def total(self) -> None:
        if self.use_batchapi:
            print("Total Cost(BatchAPI, USD): ", round(self.total_usd, 3))
            print("Total Cost(BatchAPI, JPY): ", round(self.total_jpy, 1))
        else:
            print("Total Cost(USD): ", round(self.total_usd, 3))
            print("Total Cost(JPY): ", round(self.total_jpy, 1))
        print("Number of Prompt tokens: ", self.prompt_tokens)
        print("Number of Completion tokens: ", self.completion_tokens)
