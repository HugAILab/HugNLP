import torch

"""
Transform the torch logits into probabilities.
"""
def softmax(logits):
    probs = torch.softmax(torch.from_numpy(logits).float(), -1).numpy()
    return probs
