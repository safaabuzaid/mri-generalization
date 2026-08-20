from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader
from data import BrainTumorDataset

def split_dataset (data, splitting_ratio = 0.8):
  train_size = int(len(data)*splitting_ratio)
  val_size = len(data) - train_size
  train_data, val_data = random_split(data, [train_size,val_size])
  return train_data, val_data

def create_dataset (train_path, test_path, transform=None):
  dataB_train = BrainTumorDataset(train_path,transform=None)
  dataB_test = BrainTumorDataset(test_path,transform=None)
  dataB = ConcatDataset([dataB_train,dataB_test])

  return dataB

def create_dataloaders (train, val, test, batch_size = 32):
  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader


  