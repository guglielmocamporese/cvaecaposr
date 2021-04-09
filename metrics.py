##################################################
# Imports
##################################################

import torch
import torch.nn.functional as F

def accuracy(logits, labels):
    """
    Compute the accuracy.

    Args:
        logits: tensor of shape [bs, num_classes].
        labels: tensor of shape [bs].

    Output:
        acc: scalar.
    """
    preds = F.softmax(logits, -1)
    acc = 1.0 * (preds.argmax(-1) == labels)
    return acc.mean(0)
