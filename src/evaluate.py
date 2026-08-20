import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def test(model, test_loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = sum(
        p == l for p, l in zip(all_preds, all_labels)
    ) / len(all_labels)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    return all_labels, all_preds

def create_confusion_matrix (all_labels, all_preds):
    
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Glioma","Meningioma","Pituitary"],
        yticklabels=["Glioma","Meningioma","Pituitary"]
        )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def get_classification_report (all_labels, all_preds):
    
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[
          "Glioma",
          "Meningioma",
          "Pituitary" ]
    )

    print (report)
    return report 