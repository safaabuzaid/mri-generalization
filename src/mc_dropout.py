import torch
import torch.nn as nn


def enable_mc_dropout(model):
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    return model


def mc_dropout_predict(model, images, mc_samples=30):

    predictions = []

    for _ in range(mc_samples):
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions.append(probabilities)

    predictions = torch.stack(predictions)

    mean_probabilities = predictions.mean(dim=0)

    predicted_classes = torch.argmax(
        mean_probabilities,
        dim=1
    )

    uncertainty = -torch.sum(
        mean_probabilities *
        torch.log(mean_probabilities + 1e-8),
        dim=1
    )

    return (
        mean_probabilities,
        predicted_classes,
        uncertainty
    )