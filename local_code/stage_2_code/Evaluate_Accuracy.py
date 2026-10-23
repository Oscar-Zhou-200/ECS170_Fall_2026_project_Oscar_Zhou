
"""
Evaluate a multiclass classifier using accuracy, precision,
recall, and F1-score with macro, micro, and weighted averaging.
"""

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

    def evaluate(self):
        print('evaluating performance...')

        true_y = np.asarray(self.data['true_y'])

        pred_y = self.data['pred_y']
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
