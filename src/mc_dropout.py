import torch
import torch.nn as nn
import pandas as pd


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


def run_mc_dropout(model, test_loader, device, mc_passes=30):

    model = enable_mc_dropout(model)

    results = []

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(test_loader):

            images = images.to(device)

            mean_probabilities, predicted_classes, uncertainty = (
                mc_dropout_predict(
                    model,
                    images,
                    mc_samples=mc_passes
                )
            )

            mean_probabilities = mean_probabilities.cpu().numpy()
            predicted_classes = predicted_classes.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            labels = labels.numpy()

            for i in range(len(labels)):

                results.append({
                    "image_id": batch_idx * test_loader.batch_size + i,
                    "true_class": int(labels[i]),
                    "predicted_class": int(predicted_classes[i]),
                    "prob_class_0": float(mean_probabilities[i][0]),
                    "prob_class_1": float(mean_probabilities[i][1]),
                    "prob_class_2": float(mean_probabilities[i][2]),
                    "uncertainty": float(uncertainty[i]),
                    "correct": int(labels[i] == predicted_classes[i])
                })

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(test_loader)} batches"
                )

    return pd.DataFrame(results)