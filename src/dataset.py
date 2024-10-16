import glob
import pathlib
import random

import torch
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MultilabelCarAndColorDataset(Dataset):
    def __init__(self, im_size, split, ratio=(0.65, 0.2, 0.15)):
        allowed_splits = ["train", "validate", "test"]
        assert split in allowed_splits, f"Allowed splits are {allowed_splits}, but {split} is given."
        super(MultilabelCarAndColorDataset).__init__()
        self.root_dir = pathlib.Path(r"datasets/MultilabelCarAndColorDataset")
        self.labels = ["matiz", "rio", "tiggo", "black", "blue", "red"]
        all_files = glob.glob(str(self.root_dir / "*/*.png"), recursive=True)
        all_files.extend(glob.glob(str(self.root_dir / "*/*.jpg"), recursive=True))
        split_idx = allowed_splits.index(split)
        self.im_size = im_size
        t_v_te = self._split_list(all_files, train_ratio=ratio[0], validation_ratio=ratio[1])
        self.files = t_v_te[split_idx]
        print(split, "size:", len(self))

    @staticmethod
    def _split_list(data, train_ratio, validation_ratio):
        """Splits a list into training, testing, and validation sets based on given ratios.

        Args:
            data: The list to be split.
            train_ratio: The ratio of the training set.
            validation_ratio: The ratio of the testing set.

        Returns:
            A tuple containing the training, testing, and validation sets.
        """

        if train_ratio + validation_ratio > 1:
            raise ValueError("Train and test ratios must sum to less than or equal to 1.")

        random.seed(37)
        random.shuffle(data)  # Randomly shuffle the data to ensure unbiased splitting

        train_size = int(len(data) * train_ratio)
        validate_size = int(len(data) * validation_ratio)

        training_set = data[:train_size]
        validation_set = data[train_size:train_size + validate_size]
        test_set = data[train_size + validate_size:]

        return training_set, validation_set, test_set

    def _label_encoder(self, path) -> torch.Tensor:
        return torch.tensor(list(map(lambda lbl: lbl in path, self.labels))).to(torch.float16)

    def _label_decoder(self, probabilities, threshold=0.5) -> list:
        multi_hot = (probabilities >= threshold).view((1,)).to(torch.int8)
        return list(map(lambda idx: self.labels[idx.item()], multi_hot.argwhere([1])))

    def _get_image(self, path) -> torch.Tensor:
        img = Image.open(path)
        if img.mode == "RGBA":
            # Convert RGBA to RGB mode, discarding the alpha channel
            img = img.convert("RGB")
        img = transforms.Compose([
            transforms.Resize(self.im_size),
            transforms.RandomRotation((-180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])(img)
        assert img.shape[0] == 3, path
        return img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image_file_path = self.files[item]
        return self._get_image(image_file_path), self._label_encoder(image_file_path)


class MyDataModule(LightningDataModule):
    def __init__(self, im_size, ratios=(0.65, 0.2, 0.15)):
        super(LightningDataModule).__init__()
        self.im_size = im_size
        self.ratios = ratios
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def _log_hyperparams(self):
        ...

    def setup(self, stage):
        # create datasets
        train_ds = MultilabelCarAndColorDataset(im_size=self.im_size, split="train", ratio=self.ratios)
        validate_ds = MultilabelCarAndColorDataset(im_size=self.im_size, split="validate", ratio=self.ratios)
        test_ds = MultilabelCarAndColorDataset(im_size=self.im_size, split="test", ratio=self.ratios)
        # create dataloaders
        self.train_loader = DataLoader(train_ds, batch_size=8)
        self.validate_loader = DataLoader(validate_ds, batch_size=8)
        self.test_loader = DataLoader(test_ds, batch_size=8)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.validate_loader

    def test_dataloader(self):
        return self.test_loader
