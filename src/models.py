import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import (
    efficientnet_b3, 
    EfficientNet_B3_Weights,
    ResNet18_Weights,
    DenseNet121_Weights)


def get_resnet18 ():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load ResNet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(
        model.fc.in_features,
        3
    )

    model.to(device)

    #Loss function
    criterion = nn.CrossEntropyLoss()
    #optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    return model, criterion, optimizer, device

def get_densenet121 ():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DenseNet121
    model = models.densenet121(weights = DenseNet121_Weights.DEFAULT)

    # Modify the final classification layer
    # For DenseNet, the final classification layer is model.classifier
    model.classifier = nn.Linear(
        model.classifier.in_features,
        3
    )

    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    return model, criterion, optimizer, device

def get_effecientnet_b3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load EfficientNetB3
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)

    model.to(device)

    #Loss function
    criterion = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    return model, criterion, optimizer, device