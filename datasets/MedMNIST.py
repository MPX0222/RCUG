from medmnist import DermaMNIST, OrganCMNIST, OrganSMNIST
from torchvision import transforms
import os
import json
import numpy as np


class MedMNIST:
    def __init__(self, shuffle=False, img_size=224, seed=1993) -> None:
        super().__init__()
        self.use_path = False
        self.img_size = img_size
        self.train_transform = [
            # transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((self.img_size, self.img_size)),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ]

        self.test_transform = []

        self.common_transform = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(18).tolist()
        if shuffle:
            self.class_order = np.random.permutation(len(self.class_order)).tolist()

    def download_data(self):
        DermaMNIST_train_dataset = DermaMNIST(split="train", root="/data/jiantao/Data/MedMNIST/DermaMNIST", download=True, size=224)
        DermaMNIST_val_dataset = DermaMNIST(split="val", root="/data/jiantao/Data/MedMNIST/DermaMNIST", download=True, size=224)
        DermaMNIST_test_dataset = DermaMNIST(split="test", root="/data/jiantao/Data/MedMNIST/DermaMNIST", download=True, size=224)

        OrganCMNIST_train_dataset = OrganCMNIST(split="train", root="/data/jiantao/Data/MedMNIST/OrganCMNIST", download=True,
                                    size=224)
        OrganCMNIST_val_dataset = OrganCMNIST(split="val", root="/data/jiantao/Data/MedMNIST/OrganCMNIST", download=True, size=224)
        OrganCMNIST_test_dataset = OrganCMNIST(split="test", root="/data/jiantao/Data/MedMNIST/OrganCMNIST", download=True,
                                   size=224)

        self.class_to_idx = {v: int(k) for k, v in DermaMNIST_train_dataset.info["label"].items()}
        self.class_to_idx.update({v: int(k)+7 for k, v in OrganCMNIST_train_dataset.info["label"].items()})
        print(self.class_to_idx)
        with open(os.path.join(os.path.dirname(__file__), "class_descs", "MedMNIST", "description_pool.json"), "r") as f:
            class_descs = json.load(f)
        self.class_descs = class_descs

        train_data1 = np.concatenate((DermaMNIST_train_dataset.imgs, DermaMNIST_val_dataset.imgs), axis=0)
        train_data2 = np.concatenate((np.expand_dims(OrganCMNIST_train_dataset.imgs, axis=-1).repeat(3, axis=-1),
                                      np.expand_dims(OrganCMNIST_val_dataset.imgs, axis=-1).repeat(3, axis=-1)), axis=0)
        self.train_data = np.concatenate((train_data1, train_data2), axis=0)

        train_targets1 = np.concatenate((DermaMNIST_train_dataset.labels, DermaMNIST_val_dataset.labels), axis=0)
        train_targets2 = np.concatenate((OrganCMNIST_train_dataset.labels, OrganCMNIST_val_dataset.labels), axis=0)+7
        self.train_targets = np.concatenate((train_targets1, train_targets2), axis=0)

        self.test_data = np.concatenate(
            (DermaMNIST_test_dataset.imgs, np.expand_dims(OrganCMNIST_test_dataset.imgs, axis=-1).repeat(3, axis=-1)), axis=0)

        self.test_targets = np.concatenate((DermaMNIST_test_dataset.labels, OrganCMNIST_test_dataset.labels+7), axis=0)
