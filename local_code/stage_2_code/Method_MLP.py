
"""
MLP method for 10-class handwritten digit classification.
"""

from local_code.base_class.method import method

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Method_MLP(method, nn.Module):

    data = None

    # Training settings
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Input: 784 pixels
        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()

        # Output: 10 class logits
        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 10)

        # Training history for convergence plots
        self.loss_history = []
        self.accuracy_history = []

    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        logits = self.fc_layer_3(h)
        log_probs = torch.log_softmax(logits, dim=1)
        return log_probs

    def fit(self, X, y):
        """Train the model using mini-batches."""

        # Convert pixel values to floats and normalize to [0, 1].
        X_tensor = torch.tensor(
            np.asarray(X), dtype=torch.float32
        ) / 255.0

        y_tensor = torch.tensor(
            np.asarray(y), dtype=torch.long
        )

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        loss_function = nn.NLLLoss()

        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(self.max_epoch):
            # Use PyTorch's original training-mode method.
            nn.Module.train(self, True)

            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                logits = self.forward(batch_X)
                loss = loss_function(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size

                predictions = logits.argmax(dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_size

            epoch_loss = total_loss / total
            epoch_accuracy = correct / total

            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

    def test(self, X):
        """Predict class labels for input samples."""

        X_tensor = torch.tensor(
            np.asarray(X), dtype=torch.float32
        ) / 255.0

        self.eval()

        with torch.no_grad():
            logits = self.forward(X_tensor)
            predictions = logits.argmax(dim=1)

        return predictions

    def run(self):
        print('method running...')
        print('--start training...')

        self.fit(
            self.data['train']['X'],
            self.data['train']['y']
        )

        print('--start testing...')

        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y']
        }
