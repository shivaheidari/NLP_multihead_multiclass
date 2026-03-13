import torch


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

class MultiHeadMetrics:
    def __init__(self, heads):
        self.heads = heads
        self.reset()

    def reset(self):
        self._y_true = {h: [] for h in self.heads}
        self._y_pred = {h: [] for h in self.heads}

    def update_batch(self, logits_dict, labels_dict):
        for head in self.heads:
            logits = logits_dict[head]
            labels = labels_dict[head]
            preds = torch.argmax(logits, dim=-1)

            self._y_true[head].extend(labels.cpu().tolist())
            self._y_pred[head].extend(preds.cpu().tolist())

    def compute(self):
        metrics = {}
        for head in self.heads:
            y_true = self._y_true[head]
            y_pred = self._y_pred[head]

            metrics[head] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average="macro"),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            }
        return metrics
