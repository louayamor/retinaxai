from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app.training.preprocessing import preprocess_fundus_image

CLASS_NAMES = ("ex", "he", "ma", "se")


class DDRLesionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        csv_path: Path,
        image_size: int = 384,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        img = preprocess_fundus_image(img, image_size=self.image_size)

        if self.transform is not None:
            img = self.transform(img)
        else:
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            img = to_tensor(img)

        mask_channels: list[torch.Tensor] = []
        h, w = self.image_size, self.image_size
        for cls_name in CLASS_NAMES:
            mask_path = row.get(f"{cls_name}_path", "")
            if mask_path and Path(str(mask_path)).exists():
                mask = Image.open(str(mask_path)).convert("L")
                mask = mask.resize((w, h), Image.NEAREST)
                mask_t = torch.from_numpy(np.array(mask, dtype=np.float32))
                mask_t = (mask_t > 0).float()
            else:
                mask_t = torch.zeros((h, w), dtype=torch.float32)
            mask_channels.append(mask_t)

        mask = torch.stack(mask_channels, dim=0)

        return img, mask
