from pathlib import Path

from torch.utils.data import DataLoader

from utils.dataset import CustomDataset


def load_dataset(args, transform=None, logger=None):
    """Load the dataset with optional transforms."""
    dataset = CustomDataset(
        args.data_path,
        num_samples=args.num_samples,
        subdirs_to_load=args.subdirs_to_load,
        transform=transform,
        return_path=True,
    )

    if dataset.return_path and logger:
        logger.info("Dataset will return image paths.")

    num_workers = args.num_workers
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    dataloader = DataLoader(dataset, **loader_kwargs)

    if logger:
        transform_str = "with transforms" if transform else "without transforms"
        logger.info(
            f"Dataset {args.dataset_name} loaded from {args.data_path} {transform_str}. "
            f"Batch size: {args.batch_size}, Workers: {num_workers}"
        )
    return dataset, dataloader


def get_real_class_key(dataset_path):
    """Return the first class directory under the dataset's real/ folder."""
    real_dir = Path(dataset_path) / "real"
    if not real_dir.exists():
        raise ValueError(f"Dataset structure invalid: {real_dir} not found")

    real_classes = [d.name for d in real_dir.iterdir() if d.is_dir()]
    if not real_classes:
        raise ValueError(f"No real class found in {real_dir}")

    return real_classes[0]
