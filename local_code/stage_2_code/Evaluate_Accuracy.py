
"""Evaluate a multiclass classifier on training and testing sets."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from local_code.base_class.evaluate import evaluate


class Evaluate_Accuracy(evaluate):
    data = None

    def calculate_metrics(self, true_y, pred_y):
        true_y = np.asarray(true_y)

        if hasattr(pred_y, 'detach'):
            pred_y = pred_y.detach().cpu().numpy()
        else:
            pred_y = np.asarray(pred_y)

        metrics = {
            'Accuracy': accuracy_score(true_y, pred_y),
        }

        for average in ('macro', 'micro', 'weighted'):
            metrics[f'Precision ({average})'] = precision_score(
                true_y, pred_y, average=average, zero_division=0
            )
            metrics[f'Recall ({average})'] = recall_score(
                true_y, pred_y, average=average, zero_division=0
            )
            metrics[f'F1 ({average})'] = f1_score(
                true_y, pred_y, average=average, zero_division=0
            )

        return metrics

    def evaluate(self):
        print('evaluating performance...')

        train_metrics = self.calculate_metrics(
            self.data['train_true_y'],
            self.data['train_pred_y']
        )

        test_metrics = self.calculate_metrics(
            self.data['test_true_y'],
            self.data['test_pred_y']
        )

        # Prefix metric names to distinguish the two datasets.
        metrics = {}

        for name, value in train_metrics.items():
            metrics[f'Train {name}'] = value

        for name, value in test_metrics.items():
            metrics[f'Test {name}'] = value

        return metrics
