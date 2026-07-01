import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


def evaluate_metrics(y_score, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = float(threshold)

    idx = next(
        (i for i, fpr_value in enumerate(fpr) if fpr_value > 0.05),
        len(fpr) - 1,
    )
    tpr_at_5fpr = tpr[idx - 1] if idx > 0 else 0.0

    threshold = float(0)
    y_pred = (y_score >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "TPR@5FPR": tpr_at_5fpr,
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
    }


def evaluate_model(model, test_loader, device, debug=False):
    # Set start method to spawn for CUDA multiprocessing
    if torch.cuda.is_available():
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    model = model.to(device)
    model.eval()
    y_true, y_score, class_tag, path = [], [], [], []

    with torch.no_grad():
        iterator = (
            test_loader if debug else tqdm(test_loader, desc="Evaluating", leave=False)
        )
        for batch in iterator:
            inputs, labels, tags, *img_path = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            y_score.extend(outputs.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
            class_tag.extend(tags)

            if test_loader.dataset.return_path:
                path.extend(img_path[0])

            if debug:
                break

    # Check if the model is binary or multiclass with 2 classes
    y_score = np.array(y_score)

    # Ensure y_score has a second dimension before accessing it
    if len(y_score.shape) > 1 and y_score.shape[1] == 2:
        y_score = y_score[:, 1] - y_score[:, 0]

    return (
        (np.array(y_score), np.array(y_true), class_tag, path)
        if test_loader.dataset.return_path
        else (np.array(y_score), np.array(y_true), class_tag)
    )


def result_to_dict(results):
    """Group evaluation scores by class tag."""
    result_dict = {}
    for i, tag in enumerate(results[2]):
        if tag not in result_dict:
            result_dict[tag] = []
        result_dict[tag].append((results[0][i], results[1][i], results[3][i]))
    return result_dict


def process_evaluation_scores(result_dict, real_class_key):
    """
    Process the evaluation results and return evaluation metrics.

    Args:
        result_dict: Dictionary containing results for each class
        real_class_key: Key identifying the real/authentic class in the result_dict

    Returns:
        Dictionary containing evaluation metrics for each fake class
    """
    evaluation_results = {}

    if real_class_key not in result_dict:
        raise ValueError(f"{real_class_key} not found in result_dict")

    real_result = result_dict[real_class_key]

    # Process each class except the real class
    for key, fake_result in result_dict.items():
        if key != real_class_key:
            whole_result = fake_result + real_result
            y_score, y_true, _ = zip(*whole_result)
            y_score = np.array(y_score)
            evaluation_results[key] = evaluate_metrics(y_score, y_true)

    return evaluation_results
