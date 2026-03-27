import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, test_loader, class_names, device):
    model.eval()

    true_labels = []
    preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=class_names)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    return accuracy, cm, report
