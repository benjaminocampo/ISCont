import torch
import random
import numpy as np
from collections.abc import MutableMapping
from typing import Any, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep).items()
        elif isinstance(v, list) or isinstance(v, list):
            #  For lists we transform them into strings with a join
            yield new_key, "#".join(map(str, v))
        else:
            yield new_key, v


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a dictionary using recursion (via an auxiliary funciton).
    The list/tuples values are flattened as a string.

    Parameters
    ----------
    d : MutableMapping
        Dictionary (or, more generally something that is a MutableMapping) to flatten.
        It might be nested, thus the function will traverse it to flatten it.
    parent_key : str
        Key of the parent dictionary in order to append to the path of keys.
    sep : str
        Separator to use in order to represent nested structures.

    Returns
    -------
    Dict[str, Any]
        The flattened dict where each nested dictionary is expressed as a path with
        the `sep`.

    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': {'e': {'f': 3}}})
    {'a.b': 1, 'a.c': 2, 'd.e.f': 3}
    >>> flatten_dict({'a': {'b': [1, 2]}})
    {'a.b': '1#2'}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))


def predicted_labels_from_logits(logits, threshold=0.5):
    """
    Converts logits to binary predictions using a specified threshold.

    Args:
        logits (array-like): The raw output from the model before applying any activation function.
        threshold (float): The threshold for converting probabilities to binary predictions.

    Returns:
        np.ndarray: Binary predictions based on the specified threshold.
    """
    probs = torch.softmax(torch.FloatTensor(logits), dim=1)
    pred_labels = torch.max(probs, dim=1)
    return pred_labels.indices


def multi_label_metrics(logits, labels, threshold=0.5):
    y_pred = predicted_labels_from_logits(logits=logits, threshold=threshold)
    y_true = labels

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}
    return metrics


def concatenate_columns(row, columns):
    """
    Concatenate the values of specified columns in a pandas DataFrame row into a single string,
    treating NaN values as empty strings.

    Args:
        row (pandas.Series): A single row from a pandas DataFrame.
        columns (list of str): A list of column names whose values will be concatenated.

    Returns:
        str: A single string containing the concatenated values from the specified columns of the row.
    """
    concatenated = ""
    for column in columns:
        concatenated += f"{row[column] if pd.notna(row[column]) else ''}"
    return concatenated


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
