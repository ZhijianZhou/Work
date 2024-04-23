import torch
import torch.nn as nn
from Model.utils import *
import torch.nn.functional as F

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)
def calculate_accuracy(predictions, targets, atom_mask):
    # Convert predicted probabilities to class labels
    _, predicted_labels = torch.max(predictions, 1)
    # Convert elements and atom_mask to PyTorch tensors (if they are not already)


    # Apply the atom_mask to both predicted_labels and targets
    masked_predicted_labels = predicted_labels[atom_mask.to(torch.bool)]
    masked_targets = targets[atom_mask.to(torch.bool)]

    # Compare masked predicted labels with masked ground truth labels
    correct_predictions = ((masked_predicted_labels == masked_targets)).sum().item()

    # Count the number of non-zero elements in atom_mask
    num_non_zero = masked_targets.shape[0]

    # Calculate accuracy (exclude masked zero values)
    accuracy = correct_predictions / num_non_zero
    return accuracy

def cal_atom_predict_loss(dictionary_len,predict, elements, atom_mask,criterion):
    predict = predict.view(-1, dictionary_len)
    elements = elements.view(-1)
    atom_mask = atom_mask.view(-1)
    accuracy = calculate_accuracy(predict, elements, atom_mask)
    loss = criterion(predict, elements)
    masked_loss = loss * atom_mask
    real_loss = torch.sum(masked_loss) / torch.sum(atom_mask)
    return real_loss, accuracy
