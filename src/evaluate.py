import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from argparse import Namespace
import torch
import torchvision

from utils.logger import initialize_logger
from utils.parser import get_args
from utils.detector import load_detector
from utils.transform import (
    get_detector_default_transform,
    JPEGCompression,
    AddNoise,
    ResizeByFraction,
)
from utils.eval import process_evaluation_scores, result_to_dict, evaluate_model
from utils.dataloader import load_dataset, get_real_class_key
from torchvision.transforms import v2


def save_sample_image(img, path: Path) -> bool:
    """Save a sample image for preprocessing sanity checks."""
    try:
        if hasattr(img, "permute"):
            torchvision.utils.save_image(img.clone().cpu().detach(), path)
        else:  # It's a PIL Image
            img.save(path)

        return True
    except Exception as e:
        logging.error(f"Failed to save sample image: {str(e)}")
        return False


def save_evaluation_results(evaluation_results, save_evaluation_path, logger):
    """Save evaluation metrics to JSON."""
    with open(save_evaluation_path, "w") as f:
        json.dump(evaluation_results, f)
    logger.info(f"Results saved at {save_evaluation_path}")


def save_evaluation_scores_to_csv(result_dict, save_results_path, logger):
    """Save per-sample evaluation scores to CSV."""
    with open(save_results_path, "w") as csv_file:
        fieldnames = ["class", "y_true", "y_score", "path"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in result_dict.items():
            for y_score, y_true, path in value:
                writer.writerow(
                    {"class": key, "y_true": y_true, "y_score": y_score, "path": path}
                )
    logger.info(f"Results saved at {save_results_path}")


@dataclass
class EvaluationPaths:
    """Container for evaluation output paths."""

    csv_path: Path
    json_path: Path
    sample_image_path: Path

    @classmethod
    def create(
        cls, output_dir: Path, detector_name: str, dataset_name: str
    ) -> "EvaluationPaths":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / detector_name / dataset_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            csv_path=run_dir / "results.csv",
            json_path=run_dir / "evaluation.json",
            sample_image_path=run_dir / "sample_preprocessed.png",
        )


class DetectorEvaluator:
    """Handles the evaluation of deepfake detectors with optional preprocessing."""

    def __init__(self, args: Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def create_preprocessing_transform(self):
        """Create preprocessing transform with configurable parameters."""
        return v2.Compose(
            [
                JPEGCompression(quality=self.args.jpeg_quality),
                AddNoise(mean=0, std=self.args.noise_std),
                v2.GaussianBlur(
                    kernel_size=self.args.blur_kernel, sigma=self.args.blur_std
                ),
                ResizeByFraction(fraction=self.args.resize_fraction),
            ]
        )

    def initialize_detector(self):
        """Initialize the detector model and its transforms."""
        try:
            detector = load_detector(
                self.args.detector_name,
                attack_mode=False,
                checkpoint_dir=self.args.checkpoint_dir,
                checkpoint_overrides=self.args.checkpoint,
            )

            detector_transform = get_detector_default_transform(
                self.args.detector_name,
                normalize=True,
                include_to_tensor=not self.args.preprocessing,
            )

            if self.args.preprocessing:
                preprocess_transform = self.create_preprocessing_transform()
                combined_transform = v2.Compose(
                    [
                        preprocess_transform,
                        v2.ToTensor(),
                        detector_transform,
                    ]
                )
                self.logger.info(f"Combined transform: {combined_transform}")
                return detector, combined_transform

            self.logger.info(f"Using standard transform: {detector_transform}")
            return detector, detector_transform

        except Exception as e:
            raise RuntimeError(f"Failed to initialize detector: {str(e)}") from e

    def save_results(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, Any],
        paths: EvaluationPaths,
    ) -> None:
        """Save evaluation results to files."""
        save_operations = [
            (save_evaluation_scores_to_csv, paths.csv_path, results, "CSV results"),
            (save_evaluation_results, paths.json_path, metrics, "JSON metrics"),
        ]

        for save_func, path, data, desc in save_operations:
            try:
                save_func(data, path, self.logger)
                self.logger.info(f"Successfully saved {desc}")
            except Exception as e:
                self.logger.error(f"Failed to save {desc}: {str(e)}")

    def evaluate(self) -> Dict[str, Any]:
        """Execute the full evaluation pipeline."""
        try:
            detector, transforms = self.initialize_detector()

            real_class_key = get_real_class_key(self.args.data_path)
            self.logger.info(f"Detected real class key: {real_class_key}")

            dataset, dataloader = load_dataset(self.args, transforms, self.logger)

            paths = EvaluationPaths.create(
                Path(self.args.output_dir),
                self.args.detector_name,
                self.args.dataset_name,
            )

            if self.args.preprocessing:
                try:
                    if hasattr(dataset, "data") and len(dataset.data) > 0:
                        from PIL import Image

                        orig_image_path = dataset.data[0][0]
                        original_img = Image.open(orig_image_path).convert("RGB")
                        preprocess_transform = self.create_preprocessing_transform()
                        preprocessed_img = preprocess_transform(original_img)

                        self.logger.info(
                            f"Saving sample preprocessed image to {paths.sample_image_path}"
                        )
                        if save_sample_image(preprocessed_img, paths.sample_image_path):
                            self.logger.info(
                                f"Successfully saved sample preprocessed image from {orig_image_path}"
                            )
                        self.logger.info("Preprocessing parameters:")
                        self.logger.info(f"  - JPEG Quality: {self.args.jpeg_quality}")
                        self.logger.info(
                            f"  - Resize Fraction: {self.args.resize_fraction}"
                        )
                        self.logger.info(
                            f"  - Blur Standard Deviation: {self.args.blur_std}"
                        )
                        self.logger.info(
                            f"  - Blur Kernel Size: {self.args.blur_kernel}"
                        )
                        self.logger.info(
                            f"  - Noise Standard Deviation: {self.args.noise_std}"
                        )
                    else:
                        self.logger.error(
                            "Dataset doesn't have the expected 'data' attribute or it's empty"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Failed to save sample preprocessed image: {str(e)}"
                    )

            evaluation_results = evaluate_model(
                detector, dataloader, self.device, debug=self.args.debug
            )
            result_dict = result_to_dict(evaluation_results)
            processed_metrics = process_evaluation_scores(
                result_dict, real_class_key
            )

            self.save_results(result_dict, processed_metrics, paths)

            return processed_metrics

        except Exception as error:
            self.logger.error(f"Evaluation failed: {str(error)}", exc_info=True)
            raise


def main() -> None:
    """Entry point for evaluation script."""
    args = get_args("eval")
    logger = initialize_logger(args)

    try:
        evaluator = DetectorEvaluator(args, logger)
        evaluator.evaluate()
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
