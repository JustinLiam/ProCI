import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SewerMLDataset(Dataset):

    def __init__(self, csv_file, root_dir, split='Train', transform=None, classes_names=None, binarize_targets=True):

        self.annotations_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.is_test = (self.split.lower() == 'test')
        self.classes_names = list(classes_names) if classes_names is not None else None
        self.binarize_targets = binarize_targets

        print("checking data...")
        valid_indices = []
        for idx in range(len(self.annotations_df)):
            img_filename = self.annotations_df.iloc[idx, 0]
            img_path = os.path.join(self.root_dir, self.split.capitalize(), img_filename)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                print(f"cannot find {img_path}")

        self.annotations_df = self.annotations_df.iloc[valid_indices].reset_index(drop=True)
        print(f"check done。sample num: {len(self.annotations_df)}")

        if not self.is_test:
            assert self.classes_names is not None and len(self.classes_names) > 0, \
                "Train/Val  classes_names wrong"
            miss = [c for c in self.classes_names if c not in self.annotations_df.columns]
            assert not miss, f"cannot find: {miss}"
            self.has_labels = True
        else:
            self.has_labels = False
        if not self.is_test:
            self.class_weights  = self._calculate_class_weights()

    @property
    def num_classes(self):
        return len(self.classes_names) if self.classes_names is not None else None

    def __len__(self):
        return len(self.annotations_df)

    def _calculate_class_weights(self):
        assert self.has_labels, "Test split, cannot compute class weights"
        labels_array = self.annotations_df[self.classes_names].to_numpy(dtype=np.float32)
        if self.binarize_targets:
            labels_array = (labels_array > 0).astype(np.float32)
        data_len = len(labels_array)
        pos = labels_array.sum(axis=0)
        neg = data_len - pos
        w = np.where(pos > 0, neg / pos, 1.0)
        return torch.from_numpy(w.astype(np.float32))

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        fname = row['Filename'] if 'Filename' in self.annotations_df.columns else row.iloc[0]
        img_path = os.path.join(self.root_dir, self.split.capitalize(), fname)
        image = Image.open(img_path).convert('RGB')

        if self.has_labels:
            labels = row[self.classes_names].to_numpy(dtype=np.float32, copy=True)
            if self.binarize_targets:
                labels = (labels > 0).astype(np.float32)
            labels = torch.from_numpy(labels)
        else:
            labels = None

        if self.transform:
            image = self.transform(image)

        return image, labels, img_path

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), None, []
    images  = [b[0] for b in batch]
    labels  = [b[1] for b in batch]
    fnames  = [b[2] for b in batch]
    images  = torch.utils.data.dataloader.default_collate(images)
    labels  = None if all(l is None for l in labels) else torch.utils.data.dataloader.default_collate(labels)
    return images, labels, fnames