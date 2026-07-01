import argparse
from pathlib import Path


def get_args(mode="attack"):
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks or evaluation"
    )

    # Common arguments
    parser.add_argument(
        "--detector_name",
        type=str,
        required=True,
        help="Name of the detector to evaluate",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use from dataset. If None, use all samples.",
    )
    parser.add_argument(
        "--subdirs_to_load",
        nargs="*",
        default=None,
        help="List of subdirectories to load from the dataset. If None, load all subdirs.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing detector checkpoints. Default: data/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="DETECTOR=PATH",
        help="Override one detector checkpoint path.",
    )

    if mode == "eval":
        # Preprocessing arguments
        parser.add_argument(
            "--preprocessing",
            action="store_true",
            help="Enable preprocessing transformations",
        )
        parser.add_argument(
            "--jpeg_quality",
            type=int,
            default=20,
            help="JPEG compression quality factor (0-100)",
        )
        parser.add_argument(
            "--resize_fraction",
            type=float,
            default=0.5,
            help="Fraction to scale image dimensions by (e.g., 0.5 for half size)",
        )
        parser.add_argument(
            "--blur_std",
            type=float,
            default=1.0,
            help="Standard deviation for Gaussian blur",
        )
        parser.add_argument(
            "--noise_std",
            type=float,
            default=0.1,
            help="Standard deviation for Gaussian noise as fraction (e.g., 0.1 for 10%%)",
        )
        parser.add_argument(
            "--blur_kernel",
            type=int,
            default=5,
            help="Kernel size for Gaussian blur (e.g., 5 for 5x5)",
        )
    elif mode == "attack":
        # Attack-specific arguments
        parser.add_argument(
            "--attack_type",
            type=str,
            default="pgd",
            choices=["pgd", "ensemble", "diverse", "universal", "query"],
            help="Type of attack to perform",
        )
        parser.add_argument(
            "--config_path",
            type=str,
            default="configs/attacks/attack_configs_eps_8.json",
            help="Path to attack configuration file",
        )
        parser.add_argument(
            "--ensemble_detectors",
            type=str,
            help="Comma-separated list of detector names for ensemble attacks",
        )
        parser.add_argument(
            "--skip_images", action="store_true", help="Skip saving example images"
        )

    args = parser.parse_args()

    # Convert paths to Path objects
    args.data_path = Path(args.data_path)
    args.output_dir = Path(args.output_dir)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = Path(args.checkpoint_dir)
    if mode == "attack":
        args.config_path = Path(args.config_path)

    return args
