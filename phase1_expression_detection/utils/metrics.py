import torch


def accuracy(outputs, labels):
    """
    Calculate classification accuracy

    Args:
        outputs:
            Model prediction logits
            Shape: [batch_size, num_classes]

        labels:
            Ground truth labels
            Shape: [batch_size]

    Returns:
        accuracy percentage
    """

    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc