import numpy as np
import numpy as np
import torch

def compute_class_weights(labels, num_classes):
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * counts.clip(min=1))
    return torch.tensor(weights, dtype=torch.float32)
