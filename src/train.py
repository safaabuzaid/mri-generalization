import torch 
from evaluate import validate

def train_one_epoch(model,loader,optimizer,criterion,device):
  model.train()

  running_loss = 0
  total = 0
  correct = 0

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)

    #forward pass
    outputs = model(images)

    #Compute loss
    loss = criterion(outputs, labels)

    #backpropagation
    optimizer.zero_grad()
    loss.backward()

    #update weights
    optimizer.step()

    running_loss += loss.item()

    #accuracy
    _,pred = torch.max(outputs,1)
    total += labels.size(0)
    correct += (pred == labels).sum().item()

  epoch_loss = running_loss / len(loader)
  epoch_acc = correct / total

  return epoch_loss, epoch_acc


def train_model (model, train_loader,val_loader, optimizer, criterion, device, num_epochs = 10):
    best_val_acc = 0
    best_epoch = 0

        # Store history for plotting later
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

    # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            best_train_acc = train_acc
            best_train_loss = train_loss
            best_val_loss = val_loss

            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("-" * 30)

    return (
        history,
        best_epoch,
        best_train_loss,
        best_train_acc,
        best_val_loss,
        best_val_acc
    ) 