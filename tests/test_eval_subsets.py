"""Capped inference must not submit incomplete rows to leaderboard validation."""

from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table
from relbench.base.task_entity import EntityTask
from relbench.metrics import roc_auc

from rt import eval_utils
from rt.tasks import Task


def test_capped_and_complete_predictions_use_aligned_ground_truth(monkeypatch, tmp_path):
    df = pd.DataFrame({'driverId': [0, 1, 2, 3], 'date': pd.to_datetime(['2020-01-01'] * 4),
                       'qualifying': [0, 1, 0, 1]})
    table = Table(df, {'driverId': 'drivers'}, time_col='date')
    rtask = SimpleNamespace(entity_col='driverId', time_col='date', target_col='qualifying',
                            metrics=[roc_auc], get_table=lambda *a, **k: table)
    rtask.evaluate = MethodType(EntityTask.evaluate, rtask)

    def leaderboard(name, path, dataset):
        predictions = pd.read_csv(path)
        assert len(predictions) == 4, 'leaderboard requires the full test set'
        aligned = predictions.set_index('driverId').loc[df.driverId, 'qualifying'].to_numpy()
        return {'roc_auc': roc_auc(df.qualifying.to_numpy(), aligned)}

    monkeypatch.setattr(eval_utils, '_relbench', lambda: (None, leaderboard))
    monkeypatch.setattr(eval_utils, '_load_relbench_task', lambda *a: rtask)
    monkeypatch.setattr(eval_utils, '_seed_offset', lambda *a: 100)
    monkeypatch.setattr(eval_utils, 'read_meta', lambda *a: {'source': 'test-source'})
    task = Task('rel-f1', 'driver-top3', 'qualifying', 'clf', 'test')
    # Shuffled subset: scoring must use node-index alignment rather than head(N).
    result = eval_utils._emit_and_score(tmp_path, task, 'pre', 'embed',
                                       np.array([1, 0]), np.array([2, -2]), np.array([103, 100]),
                                       keep_csv=True)
    assert result[:3] == ('subset_roc_auc', 1.0, 2)
    assert result[4].name == 'rel-f1__driver-top3.partial.csv'
    assert pd.read_csv(result[4]).driverId.tolist() == [3, 0]
    full = eval_utils._emit_and_score(tmp_path, task, 'pre', 'embed',
                                     np.array([1, 0, 1, 0]), np.array([2, -2, 2, -2]),
                                     np.array([103, 100, 101, 102]), keep_csv=True)
    assert full[:3] == ('roc_auc', 1.0, 4)
    assert full[4].name == 'rel-f1__driver-top3.csv'
    with pytest.raises(RuntimeError, match='duplicate'):
        eval_utils._emit_and_score(tmp_path, task, 'pre', 'embed',
                                  [0, 0], [0, 0], [100, 100], keep_csv=False)


def test_single_class_smoke_metric_is_unavailable():
    name, value = eval_utils.metric_for('clf', np.array([1, 1]), np.array([0.2, 0.5]))
    assert name == 'roc_auc'
    assert np.isnan(value)
