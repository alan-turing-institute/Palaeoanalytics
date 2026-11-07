"""
Shared training code for the package's random forests.

The scale bar and arrow training tools fit a scikit-learn forest, score
every row with a forest that never saw its group (a page or a crop),
and export the trees as the parallel node arrays that
``pylithics.forest`` reads. This module holds those three steps.
"""

import json
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Settings = Tuple[int, int, int]     # trees, depth, smallest leaf


def fit(x: np.ndarray, y: np.ndarray, trees: int, depth: int, leaf: int):
    """Fit a balanced random forest; return it."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
        class_weight='balanced', random_state=0, n_jobs=-1,
    )
    model.fit(x, y)
    return model


def held_out_scores(
    x: np.ndarray, y: np.ndarray, groups: Sequence[str], settings: Settings,
    folds: Optional[int] = None,
) -> np.ndarray:
    """
    Score every row with a forest trained without its group.

    With ``folds`` the groups are dealt into that many folds (grouped
    k-fold); without, each group is held out on its own.
    """
    groups = np.array(groups)
    names = sorted(set(groups))
    if folds:
        fold_of = {name: i % folds for i, name in enumerate(names)}
        parts = [[n for n in names if fold_of[n] == f] for f in range(folds)]
    else:
        parts = [[n] for n in names]
    scores = np.zeros(len(y))
    for part in parts:
        held = np.isin(groups, part)
        if held.all() or len(set(y[~held])) < 2:
            continue
        model = fit(x[~held], y[~held], *settings)
        scores[held] = model.predict_proba(x[held])[:, 1]
    return scores


def tree_arrays(estimator) -> Dict:
    """Export one fitted tree as the parallel node arrays the package reads."""
    tree = estimator.tree_
    counts = tree.value[:, 0, :]
    share = counts[:, 1] / np.maximum(1e-9, counts.sum(axis=1))
    return {
        'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
        'feature': np.where(tree.children_left == -1, 0, tree.feature).tolist(),
        'threshold': np.round(tree.threshold, 5).tolist(),
        'value': np.round(share, 4).tolist(),
    }


def write_model(
    path: str, features: List[str], forest, threshold: float, extra: Dict
) -> None:
    """Write the forest as JSON with its features, threshold and provenance."""
    model = {
        'kind': 'forest', 'features': list(features),
        'trees': [tree_arrays(t) for t in forest.estimators_],
        'threshold': threshold, 'trained': date.today().isoformat(),
    }
    model.update(extra)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(model, handle, separators=(',', ':'))


def print_importances(features: Sequence[str], forest) -> None:
    """Print the features by their weight in the forest."""
    pairs = sorted(zip(features, forest.feature_importances_), key=lambda p: -p[1])
    for name, weight in pairs:
        print(f"  {name:<16} {weight:.3f}")
