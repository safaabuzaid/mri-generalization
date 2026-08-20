from torch.utils.data import Dataset
from PIL import Image
import os


class BrainTumorDataset (Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        image_path = self.dataframe.loc[idx, "path"]
        label = int(self.dataframe.loc[idx, "label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label