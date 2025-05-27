from typing import Any

import wandb


def calculate_metrics(num_corrects: int, num_preds: int, num_golds: int) -> tuple[float, float, float]:
    precision = num_corrects / num_preds if num_preds > 0 else float("nan")
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return precision, recall, f1


def evaluate(predictions: list[dict[str, Any]]) -> dict[str, float]:
    n_correct, n_pred, n_gold = 0, 0, 0
    for prediction in predictions:
        golds = set(prediction['golds'])
        preds = set(prediction['preds'])
        n_gold += len(golds)
        n_pred += len(preds)
        n_correct += len(preds.intersection(golds))

    precision, recall, f1 = calculate_metrics(n_correct, n_pred, n_gold)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def submit_wandb_evaluate(metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        wandb.log({k: v})
