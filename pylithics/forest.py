"""
Random forests read from JSON and scored with numpy.

The scale bar classifier and the arrow classifier are random forests
trained with scikit-learn by the tools under ``tools/`` and exported as
JSON. This module reads such a file and scores a feature dictionary
with it, so the package needs no scikit-learn at run time.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


@dataclass
class Tree:
    """One decision tree as parallel node arrays; a leaf has ``left == -1``."""

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray


@dataclass
class Model:
    """A random forest over the named features: the mean of its trees' leaf values."""

    features: List[str]
    trees: List[Tree]


class ModelUnavailable(RuntimeError):
    """The model file is missing or unreadable."""


def load_forest(path: str, search_dir: Optional[str] = None) -> Model:
    """
    Read a forest from JSON.

    A bare file name is looked for in ``search_dir`` (by default the
    package's ``models`` folder) when it is not a file as given.

    Raises
    ------
    ModelUnavailable
        When the file is missing or does not hold a complete model.
    """
    if not os.path.isabs(path) and not os.path.isfile(path):
        path = os.path.join(search_dir or MODELS_DIR, path)
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        model = Model(
            features=list(data['features']),
            trees=[_tree_from(t) for t in data['trees']],
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailable(f"Model not read from {path}: {exc}") from exc
    if not model.trees or any(
        (t.feature >= len(model.features)).any() for t in model.trees
    ):
        raise ModelUnavailable(f"Model at {path} is not consistent")
    return model


def _tree_from(data: Dict) -> Tree:
    """Build a tree from its JSON node arrays, checking they agree in length."""
    arrays = {k: np.asarray(data[k]) for k in ('left', 'right', 'feature', 'threshold', 'value')}
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError('tree arrays differ in length')
    return Tree(
        left=arrays['left'].astype(int), right=arrays['right'].astype(int),
        feature=arrays['feature'].astype(int), threshold=arrays['threshold'].astype(float),
        value=arrays['value'].astype(float),
    )


def score_features(features: Dict[str, float], model: Model) -> float:
    """Return the forest's probability for these features; a missing feature counts 0."""
    values = np.array([features.get(name, 0.0) for name in model.features], dtype=float)
    return float(np.mean([_leaf_value(tree, values) for tree in model.trees]))


def _leaf_value(tree: Tree, values: np.ndarray) -> float:
    """Follow one tree down to its leaf for these feature values."""
    node = 0
    while tree.left[node] != -1:
        if values[tree.feature[node]] <= tree.threshold[node]:
            node = tree.left[node]
        else:
            node = tree.right[node]
    return float(tree.value[node])
