import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """Custom dataset for loading real and fake images."""

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        num_samples: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[transforms.Compose] = None,
        subdirs_to_load: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_samples = num_samples
        self.return_path = return_path
        self.transform = transform
        self.labels = {"fake": 1, "real": 0}
        self.subdirs_to_load = subdirs_to_load

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory {root_dir} does not exist")

        self.data = self._load_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int, str], Tuple[torch.Tensor, int, str, str]]:
        img_path, label, subdir = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")

        if self.transform:
            image = self.transform(image)

        return (
            (image, label, subdir, img_path)
            if self.return_path
            else (image, label, subdir)
        )

    def _get_image_paths(self, directory: Path) -> List[Path]:
        """Get all valid image paths from a directory."""
        return [
            p
            for p in directory.glob("*")
            if p.suffix.lower() in self.VALID_EXTENSIONS
            and not p.stem.endswith("_difference")
            and not p.stem.endswith("_original")
        ]

    def _load_data(self) -> List[Tuple[Path, int, str]]:
        """Load dataset paths and labels."""
        random.seed(42)
        data = []

        for label in self.labels:
            label_dir = self.root_dir / label
            if not label_dir.exists():
                continue

            subdirs = self.subdirs_to_load or [
                d.name for d in label_dir.iterdir() if d.is_dir()
            ]

            for subdir in subdirs:
                subdir_path = label_dir / subdir
                if not subdir_path.is_dir():
                    continue

                images = self._get_image_paths(subdir_path)
                if self.num_samples and len(images) > self.num_samples:
                    images = random.sample(images, self.num_samples)

                data.extend((str(img), self.labels[label], subdir) for img in images)

        if not data:
            raise RuntimeError(f"No valid images found in {self.root_dir}")

        return data
