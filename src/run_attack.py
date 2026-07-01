import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from argparse import Namespace
import torch
import torchvision

from utils.logger import initialize_logger
from utils.parser import get_args
from utils.detector import load_detector
from utils.transform import get_detector_default_transform
from utils.dataloader import load_dataset
from attacks.attacks import (
    PGDAttack,
    EnsembleAttack,
    DiverseScaleAttack,
    UniversalAttack,
    QueryEfficientAttack,
)


@dataclass
class AttackResults:
    """Container for attack results and paths."""

    metrics_path: Path
    images_path: Path
    attack_type: str
    target_detector: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    ensemble_detectors: Optional[List[str]] = None
    results_csv: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert AttackResults to a JSON serializable dictionary."""
        return {
            "metrics_path": str(self.metrics_path),
            "images_path": str(self.images_path),
            "results_csv": str(self.results_csv) if self.results_csv else None,
            "attack_type": self.attack_type,
            "target_detector": self.target_detector,
            "config": self.config,
            "metrics": self.metrics,
            "ensemble_detectors": self.ensemble_detectors,
        }

    def save_csv_results(self, results_data: List[Dict[str, Any]]) -> None:
        """Save detailed results in CSV format."""
        if results_data:
            csv_path = self.metrics_path.parent / "results.csv"
            self.results_csv = csv_path

            # Collect all possible headers from the data
            headers = set()
            for entry in results_data:
                headers.update(entry.keys())
            headers = sorted(list(headers))

            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for row in results_data:
                    writer.writerow(row)

        self.save_metrics()

    def save_metrics(self) -> None:
        """Save run metadata and metrics as JSON."""
        with open(self.metrics_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def create(
        cls,
        output_dir: Path,
        attack_type: str,
        target_detector: str,
        dataset_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        ensemble_detectors: Optional[List[str]] = None,
    ) -> "AttackResults":
        """Create standardized output paths and results container."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / attack_type / target_detector / dataset_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            metrics_path=run_dir / "metrics.json",
            images_path=run_dir / "images",
            attack_type=attack_type,
            target_detector=target_detector,
            config=config,
            metrics=metrics,
            ensemble_detectors=ensemble_detectors,
        )


class AttackRunner:
    """Handles running of adversarial attacks"""

    def __init__(self, args: Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def get_normalization_params(self, dataloader):
        """Extract normalization parameters from the dataset transform."""
        transform = getattr(dataloader.dataset, "transform", None)
        for t in getattr(transform, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                mean = [float(v) for v in t.mean]
                std = [float(v) for v in t.std]
                return mean, std

        self.logger.warning(
            "No normalization found in transform, using ImageNet defaults"
        )
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def denormalize_batch(self, images, mean, std):
        """Convert a normalized image batch back to [0, 1] image space."""
        mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
        return images * std_t + mean_t

    def initialize_detector(self):
        """Initialize detector model and transforms"""
        try:
            detector = load_detector(
                self.args.detector_name,
                attack_mode=True,
                checkpoint_dir=self.args.checkpoint_dir,
                checkpoint_overrides=self.args.checkpoint,
            )
            detector = detector.to(self.device)
            detector.eval()
            transforms = get_detector_default_transform(self.args.detector_name)
            return detector, transforms
        except Exception as e:
            raise RuntimeError(f"Failed to initialize detector: {str(e)}") from e

    def load_attack_config(self) -> Dict[str, Any]:
        """Load attack configuration from JSON file"""
        try:
            with open(self.args.config_path) as f:
                configs = json.load(f)
            return configs[self.args.attack_type]
        except Exception as e:
            raise RuntimeError(f"Failed to load attack config: {str(e)}") from e

    def get_image_output_paths(
        self, result: AttackResults, image_path: str, subdir: str
    ) -> tuple[str, Path, Path, Path]:
        """Return the image base name and output paths for one result row."""
        source_path = Path(image_path)
        save_dir = result.images_path / source_path.parent.parts[-2] / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        base_name = source_path.stem
        save_path = save_dir / base_name
        return (
            base_name,
            Path(f"{save_path}_original.png"),
            Path(f"{save_path}_adversarial.png"),
            Path(f"{save_path}_difference.png"),
        )

    def csv_image_paths(
        self, result: AttackResults, orig_path: Path, adv_path: Path, diff_path: Path
    ) -> Dict[str, str]:
        """Return image paths for CSV output."""
        if self.args.skip_images:
            return {"path_original": "", "path_adv": "", "path_diff": ""}

        run_dir = result.metrics_path.parent
        return {
            "path_original": str(orig_path.relative_to(run_dir)),
            "path_adv": str(adv_path.relative_to(run_dir)),
            "path_diff": str(diff_path.relative_to(run_dir)),
        }

    def run_pgd_attack(self, detector, dataloader, config) -> AttackResults:
        """Run PGD attack on a single detector."""
        mean, std = self.get_normalization_params(dataloader)
        attack = PGDAttack(detector, mean=mean, std=std, **config)

        metrics = {
            "original_scores": [],
            "adversarial_scores": [],
            "perturbation_stats": [],
        }

        # For CSV export
        csv_results = []

        # Create AttackResults first to get the timestamp directory
        results = AttackResults.create(
            Path(self.args.output_dir),
            "pgd",
            self.args.detector_name,
            self.args.dataset_name,
            config,
            metrics,
        )

        results.images_path.mkdir(parents=True, exist_ok=True)

        for batch_idx, (images, labels, subdirs, image_paths) in enumerate(dataloader):
            images = images.to(self.device)
            target_is_fake = (labels == 0).to(self.device)

            # Generate adversarial examples
            adv_images = attack.attack(images, target_is_fake)

            # Compute metrics
            with torch.no_grad():
                orig_scores = attack.get_fakeness_score(detector, images)
                adv_scores = attack.get_fakeness_score(detector, adv_images)
                orig_denorm_batch = self.denormalize_batch(images, mean, std).clamp(
                    0, 1
                )
                adv_denorm_batch = self.denormalize_batch(adv_images, mean, std).clamp(
                    0, 1
                )
                perturbation = (adv_denorm_batch - orig_denorm_batch).abs()

                metrics["original_scores"].extend(orig_scores.cpu().tolist())
                metrics["adversarial_scores"].extend(adv_scores.cpu().tolist())
                metrics["perturbation_stats"].append(
                    {
                        "max": perturbation.max().item(),
                        "mean": perturbation.mean().item(),
                        "l2_norm": perturbation.norm(2).item(),
                    }
                )

                # Save images and collect CSV data
                for i in range(images.size(0)):
                    orig_denorm = orig_denorm_batch[i]
                    adv_denorm = adv_denorm_batch[i]

                    base_name, orig_path, adv_path, diff_path = (
                        self.get_image_output_paths(results, image_paths[i], subdirs[i])
                    )

                    if not self.args.skip_images:
                        torchvision.utils.save_image(orig_denorm.clamp(0, 1), orig_path)
                        torchvision.utils.save_image(adv_denorm.clamp(0, 1), adv_path)
                        torchvision.utils.save_image(perturbation[i].clamp(0, 1), diff_path)

                    # Collect data for CSV with paths relative to the metrics file and 4 decimal precision
                    csv_results.append(
                        {
                            # Basic info
                            "attack_type": "pgd",
                            "detector": self.args.detector_name,
                            "dataset": self.args.dataset_name,
                            # Image info
                            "image_name": base_name,
                            "subdir": subdirs[i],
                            "label": labels[i].item(),
                            # Scores
                            "normal_score": f"{orig_scores[i].item():.4f}",
                            "adv_score": f"{adv_scores[i].item():.4f}",
                            # Perturbation metrics
                            "perturbation_l2": f"{perturbation[i].norm(2).item():.4f}",
                            "perturbation_max": f"{perturbation[i].max().item():.4f}",
                            "perturbation_mean": f"{perturbation[i].mean().item():.4f}",
                            **self.csv_image_paths(results, orig_path, adv_path, diff_path),
                        }
                    )

                # logging
                self.logger.info(
                    f"Batch {batch_idx}: Original scores: {orig_scores}, Adversarial scores: {adv_scores}"
                )

            if self.args.debug and batch_idx >= 2:
                break

            # Clean up GPU memory
            del images, adv_images, orig_scores, adv_scores, perturbation
            torch.cuda.empty_cache()

        # Update metrics in results object
        results.metrics = metrics

        # Save CSV results
        results.save_csv_results(csv_results)

        return results

    def run_ensemble_attack(self, detectors, dataloader, config) -> List[AttackResults]:
        """Run ensemble attack using all detectors in ensemble."""
        results = []

        mean, std = self.get_normalization_params(dataloader)

        # Get transforms for each detector
        detector_transforms = []
        for name in detectors.keys():
            transform = get_detector_default_transform(name)
            detector_transforms.append(transform)

        # Create ensemble attack with all detectors, their transforms, and normalization params
        attack = EnsembleAttack(
            models=list(detectors.values()),
            model_transforms=detector_transforms,
            mean=mean,
            std=std,
            **config,
        )

        metrics = {
            "original_scores": [],
            "adversarial_scores": [],
            "perturbation_stats": [],
        }

        csv_results = []

        # Create results for ensemble
        result = AttackResults.create(
            Path(self.args.output_dir),
            "ensemble",
            self.args.detector_name,
            self.args.dataset_name,
            config,
            metrics,
            list(detectors.keys()),  # Store all detector names for reference
        )

        result.images_path.mkdir(parents=True, exist_ok=True)

        for batch_idx, (images, labels, subdirs, image_paths) in enumerate(dataloader):
            images = images.to(self.device)
            target_is_fake = (labels == 0).to(self.device)

            # Generate adversarial examples using ensemble
            adv_images = attack.attack(images, target_is_fake)

            with torch.no_grad():
                orig_scores = attack.get_ensemble_score(images)
                adv_scores = attack.get_ensemble_score(adv_images)
                orig_denorm_batch = self.denormalize_batch(images, mean, std).clamp(
                    0, 1
                )
                adv_denorm_batch = self.denormalize_batch(adv_images, mean, std).clamp(
                    0, 1
                )
                perturbation = (adv_denorm_batch - orig_denorm_batch).abs()

                metrics["original_scores"].extend(orig_scores.cpu().tolist())
                metrics["adversarial_scores"].extend(adv_scores.cpu().tolist())
                metrics["perturbation_stats"].append(
                    {
                        "max": perturbation.max().item(),
                        "mean": perturbation.mean().item(),
                        "l2_norm": perturbation.norm(2).item(),
                    }
                )

                # Save images and collect CSV data
                for i in range(images.size(0)):
                    orig_denorm = orig_denorm_batch[i]
                    adv_denorm = adv_denorm_batch[i]
                    pert_denorm = perturbation[i]

                    base_name, orig_path, adv_path, diff_path = (
                        self.get_image_output_paths(result, image_paths[i], subdirs[i])
                    )

                    if not self.args.skip_images:
                        torchvision.utils.save_image(orig_denorm.clamp(0, 1), orig_path)
                        torchvision.utils.save_image(adv_denorm.clamp(0, 1), adv_path)
                        torchvision.utils.save_image(pert_denorm.clamp(0, 1), diff_path)

                    # Collect data for CSV with paths relative to the metrics file
                    csv_results.append(
                        {
                            "attack_type": "ensemble",
                            "dataset": self.args.dataset_name,
                            "image_name": base_name,
                            "subdir": subdirs[i],
                            "label": labels[i].item(),
                            "normal_score": f"{orig_scores[i].item():.4f}",
                            "adv_score": f"{adv_scores[i].item():.4f}",
                            "perturbation_l2": f"{perturbation[i].norm(2).item():.4f}",
                            "perturbation_max": f"{perturbation[i].max().item():.4f}",
                            "perturbation_mean": f"{perturbation[i].mean().item():.4f}",
                            "ensemble_detectors": ",".join(
                                [name for name in detectors.keys()]
                            ),
                            **self.csv_image_paths(result, orig_path, adv_path, diff_path),
                        }
                    )

                if self.args.debug and batch_idx >= 2:
                    break

                # Clean up GPU memory
                del images, adv_images, orig_scores, adv_scores, perturbation
                torch.cuda.empty_cache()

        # Save CSV results before appending this result
        result.save_csv_results(csv_results)
        results.append(result)

        return results

    def run_diverse_scale_attack(
        self, detectors, dataloader, config
    ) -> List[AttackResults]:
        """Run diverse scale attack similar to ensemble but with scale invariance."""
        results = []

        mean, std = self.get_normalization_params(dataloader)

        # Get transforms for each detector
        detector_transforms = []
        for name in detectors.keys():
            transform = get_detector_default_transform(name)
            detector_transforms.append(transform)

        # Create attack instance with merged config
        config_with_defaults = {
            "models": list(detectors.values()),
            "model_transforms": detector_transforms,  # Add model transforms
            "mean": mean,
            "std": std,
        }
        config_with_defaults.update(
            config
        )  # Config will override any defaults if present

        attack = DiverseScaleAttack(**config_with_defaults)

        # Initialize metrics
        csv_results = []
        metrics = {
            "original_scores": [],
            "adversarial_scores": [],
            "perturbation_stats": [],
        }

        # Create results container
        result = AttackResults.create(
            Path(self.args.output_dir),
            "diverse",
            self.args.detector_name,
            self.args.dataset_name,
            config,
            metrics,
            list(detectors.keys()),  # Store detector names for reference
        )

        result.images_path.mkdir(parents=True, exist_ok=True)

        for batch_idx, (images, labels, subdirs, image_paths) in enumerate(dataloader):
            images = images.to(self.device)
            target_is_fake = (labels == 0).to(self.device)

            # Generate adversarial examples with scale diversity
            adv_images = attack.attack(images, target_is_fake)

            with torch.no_grad():
                orig_scores = attack.get_ensemble_score(images)
                adv_scores = attack.get_ensemble_score(adv_images)
                orig_denorm_batch = self.denormalize_batch(images, mean, std).clamp(
                    0, 1
                )
                adv_denorm_batch = self.denormalize_batch(adv_images, mean, std).clamp(
                    0, 1
                )
                perturbation = (adv_denorm_batch - orig_denorm_batch).abs()

                # Update metrics
                metrics["original_scores"].extend(orig_scores.cpu().tolist())
                metrics["adversarial_scores"].extend(adv_scores.cpu().tolist())
                metrics["perturbation_stats"].append(
                    {
                        "max": perturbation.max().item(),
                        "mean": perturbation.mean().item(),
                        "l2_norm": perturbation.norm(2).item(),
                    }
                )

                # Save images and collect CSV data
                for i in range(images.size(0)):
                    orig_denorm = orig_denorm_batch[i]
                    adv_denorm = adv_denorm_batch[i]

                    base_name, orig_path, adv_path, diff_path = (
                        self.get_image_output_paths(result, image_paths[i], subdirs[i])
                    )

                    if not self.args.skip_images:
                        torchvision.utils.save_image(orig_denorm.clamp(0, 1), orig_path)
                        torchvision.utils.save_image(adv_denorm.clamp(0, 1), adv_path)
                        torchvision.utils.save_image(perturbation[i].clamp(0, 1), diff_path)

                    # Collect data for CSV
                    csv_results.append(
                        {
                            "attack_type": "diverse_scale",
                            "dataset": self.args.dataset_name,
                            "image_name": base_name,
                            "subdir": subdirs[i],
                            "label": labels[i].item(),
                            "normal_score": f"{orig_scores[i].item():.4f}",
                            "adv_score": f"{adv_scores[i].item():.4f}",
                            "perturbation_l2": f"{perturbation[i].norm(2).item():.4f}",
                            "perturbation_max": f"{perturbation[i].max().item():.4f}",
                            "perturbation_mean": f"{perturbation[i].mean().item():.4f}",
                            "detectors": ",".join(detectors.keys()),
                            **self.csv_image_paths(result, orig_path, adv_path, diff_path),
                        }
                    )

            if self.args.debug and batch_idx >= 2:
                break

            # Clean up GPU memory
            del images, adv_images, orig_scores, adv_scores, perturbation
            torch.cuda.empty_cache()

        # Save CSV results
        result.save_csv_results(csv_results)
        results.append(result)

        return results

    def run_query_efficient_attack(
        self, detectors, dataloader, config
    ) -> List[AttackResults]:
        """Run query-efficient black-box attack on a single detector."""
        results = []
        mean, std = self.get_normalization_params(dataloader)

        if isinstance(detectors, dict):
            det_name = list(detectors.keys())[0]
            detector = detectors[det_name]
        else:
            detector = detectors
            det_name = self.args.detector_name

        query_attack_params = {
            "model": detector,
            "eps": config.get("eps", 8 / 255),
            "max_queries": config.get("max_queries", 100),
            "samples_per_draw": config.get("samples_per_draw", 10),
            "noise_scale": config.get("noise_scale", 0.005),
            "min_step_factor": config.get("min_step_factor", 0.01),
            "mean": config.get("mean", mean),
            "std": config.get("std", std),
        }

        attack = QueryEfficientAttack(**query_attack_params)

        metrics = {
            "original_scores": [],
            "adversarial_scores": [],
            "perturbation_stats": [],
            "query_counts": [],
        }

        csv_results = []

        result = AttackResults.create(
            output_dir=Path(self.args.output_dir),
            attack_type="query",
            target_detector=det_name,
            dataset_name=self.args.dataset_name,
            config=config,
            metrics=metrics,
        )

        result.images_path.mkdir(parents=True, exist_ok=True)

        for batch_idx, (images, labels, subdirs, image_paths) in enumerate(dataloader):
            images = images.to(self.device)
            # Target the opposite class: real images -> fake, fake images -> real.
            target_is_fake = (labels == 0).to(self.device)
            attack.query_count = 0  # Reset query count for each batch

            # Generate adversarial examples with query efficiency
            adv_images = attack.attack(images, target_is_fake)

            with torch.no_grad():
                orig_scores = attack.get_fakeness_score(detector, images)
                adv_scores = attack.get_fakeness_score(detector, adv_images)
                orig_denorm_batch = attack.denormalize(images).clamp(0, 1)
                adv_denorm_batch = attack.denormalize(adv_images).clamp(0, 1)
                perturbation = (adv_denorm_batch - orig_denorm_batch).abs()

                metrics["original_scores"].extend(orig_scores.cpu().tolist())
                metrics["adversarial_scores"].extend(adv_scores.cpu().tolist())
                metrics["perturbation_stats"].append(
                    {
                        "max": perturbation.max().item(),
                        "mean": perturbation.mean().item(),
                        "l2_norm": perturbation.norm(2).item(),
                    }
                )
                metrics["query_counts"].append(attack.query_count)

                # Save images and collect data for CSV
                for i in range(images.size(0)):
                    base_name, orig_path, adv_path, diff_path = (
                        self.get_image_output_paths(result, image_paths[i], subdirs[i])
                    )
                    orig_img = orig_denorm_batch[i].cpu()
                    adv_img = adv_denorm_batch[i].cpu()

                    if not self.args.skip_images:
                        torchvision.utils.save_image(orig_img.clamp(0, 1), orig_path)
                        torchvision.utils.save_image(adv_img.clamp(0, 1), adv_path)
                        torchvision.utils.save_image(perturbation[i].cpu(), diff_path)

                    # Track target direction for reporting
                    target_direction = "decrease" if target_is_fake[i] else "increase"
                    target_value = 0.0 if target_is_fake[i] else 1.0

                    # Collect data for CSV with paths relative to the metrics file
                    csv_results.append(
                        {
                            "attack_type": "query_efficient",
                            "detector": det_name,
                            "dataset": self.args.dataset_name,
                            "image_name": base_name,
                            "subdir": subdirs[i],
                            "label": labels[i].item(),
                            "target_is_fake": int(target_is_fake[i].item()),
                            "target_direction": target_direction,
                            "target_value": f"{target_value:.1f}",
                            "normal_score": f"{orig_scores[i].item():.4f}",
                            "adv_score": f"{adv_scores[i].item():.4f}",
                            "score_change": f"{(adv_scores[i] - orig_scores[i]).item():.4f}",
                            "perturbation_l2": f"{perturbation[i].norm(2).item():.4f}",
                            "perturbation_max": f"{perturbation[i].max().item():.4f}",
                            "perturbation_mean": f"{perturbation[i].mean().item():.4f}",
                            "query_count": attack.query_count,
                            **self.csv_image_paths(result, orig_path, adv_path, diff_path),
                        }
                    )

            if self.args.debug and batch_idx >= 2:
                break

            # Clean up GPU memory
            del images, adv_images, orig_scores, adv_scores, perturbation
            torch.cuda.empty_cache()

        # Update metrics in results object
        result.metrics = metrics

        # Save CSV results
        result.save_csv_results(csv_results)
        results.append(result)

        return results

    def run_universal_attack(
        self, detectors, dataloader, config
    ) -> List[AttackResults]:
        """Run universal attack using all detectors in ensemble, with dual perturbations."""
        results = []

        mean, std = self.get_normalization_params(dataloader)

        # Get transforms for each detector
        detector_transforms = []
        for name in detectors.keys():
            transform = get_detector_default_transform(name)
            detector_transforms.append(transform)

        # Create universal attack with all detectors and enable dual perturbation
        config_with_defaults = config.copy()
        config_with_defaults["dual_perturbation"] = True  # Enable dual perturbation

        attack = UniversalAttack(
            models=list(detectors.values()),
            model_transforms=detector_transforms,
            mean=mean,
            std=std,
            **config_with_defaults,
        )

        # Initialize metrics
        metrics = {
            "original_scores": [],
            "adversarial_scores": {
                "fake_to_real": [],  # Scores when applying fake→real perturbation
                "real_to_fake": [],  # Scores when applying real→fake perturbation
            },
            "perturbation_stats": {
                "fake_to_real": {},  # Stats for fake→real perturbation
                "real_to_fake": {},  # Stats for real→fake perturbation
            },
        }

        csv_results = []

        # Create attack results container
        result = AttackResults.create(
            Path(self.args.output_dir),
            "universal",  # Changed name to reflect dual perturbation
            self.args.detector_name,
            self.args.dataset_name,
            config_with_defaults,
            metrics,
            list(detectors.keys()),
        )

        result.images_path.mkdir(parents=True, exist_ok=True)

        # Generate dual universal perturbations
        self.logger.info("Generating dual universal perturbations...")
        perturbations = attack.generate_perturbation(dataloader)

        # Extract the two perturbations
        if isinstance(perturbations, dict):
            delta_fake_to_real = perturbations["fake_to_real"]
            delta_real_to_fake = perturbations["real_to_fake"]
        else:
            self.logger.warning("Only a single perturbation was generated")
            delta_fake_to_real = perturbations
            delta_real_to_fake = perturbations

        # Track perturbation statistics
        if delta_fake_to_real is not None:
            metrics["perturbation_stats"]["fake_to_real"] = {
                "magnitude": torch.norm(delta_fake_to_real).item(),
                "max": delta_fake_to_real.abs().max().item(),
                "mean": delta_fake_to_real.abs().mean().item(),
            }

        if delta_real_to_fake is not None:
            metrics["perturbation_stats"]["real_to_fake"] = {
                "magnitude": torch.norm(delta_real_to_fake).item(),
                "max": delta_real_to_fake.abs().max().item(),
                "mean": delta_real_to_fake.abs().mean().item(),
            }

        # Apply universal perturbations to all images
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                images, labels, subdirs, image_paths = batch
            else:
                images = batch
                subdirs = ["unknown"] * images.size(0)
                image_paths = [f"img_{i}" for i in range(images.size(0))]

            images = images.to(self.device)

            # Get original scores
            orig_scores = attack.get_ensemble_score(images)

            # Split images into real and fake groups
            real_indices = [i for i, label in enumerate(labels) if label == 0]
            fake_indices = [i for i, label in enumerate(labels) if label == 1]

            # Get real and fake images separately if present
            real_images = images[real_indices] if real_indices else None
            fake_images = images[fake_indices] if fake_indices else None

            # Apply appropriate perturbation to each category
            # 1. Apply real→fake perturbation to real images
            if real_images is not None:
                # Set target_is_fake=True for real→fake transformation
                adv_real = attack.attack(real_images, target_is_fake=True)
                adv_real_scores = attack.get_ensemble_score(adv_real)
                metrics["adversarial_scores"]["real_to_fake"].extend(
                    adv_real_scores.cpu().tolist()
                )

            # 2. Apply fake→real perturbation to fake images
            if fake_images is not None:
                # Set target_is_fake=False for fake→real transformation
                adv_fake = attack.attack(fake_images, target_is_fake=False)
                adv_fake_scores = attack.get_ensemble_score(adv_fake)
                metrics["adversarial_scores"]["fake_to_real"].extend(
                    adv_fake_scores.cpu().tolist()
                )

            # Update original scores tracking
            metrics["original_scores"].extend(orig_scores.cpu().tolist())

            # Save images and collect CSV data
            for idx, i in enumerate(range(images.size(0))):
                # Determine which perturbation to use based on image label
                if i in real_indices:
                    adv_images_i = adv_real[real_indices.index(i)]
                    perturbation_type = "real_to_fake"
                    adv_score = adv_real_scores[real_indices.index(i)].item()
                else:
                    adv_images_i = adv_fake[fake_indices.index(i)]
                    perturbation_type = "fake_to_real"
                    adv_score = adv_fake_scores[fake_indices.index(i)].item()

                # Calculate perturbation for this image
                perturbation = (
                    attack.denormalize(adv_images_i).unsqueeze(0)
                    - attack.denormalize(images[i]).unsqueeze(0)
                ).abs()

                base_name, orig_path, adv_path, diff_path = self.get_image_output_paths(
                    result, image_paths[i], subdirs[i]
                )

                if not self.args.skip_images:
                    torchvision.utils.save_image(
                        attack.denormalize(images[i]).cpu(), orig_path
                    )
                    torchvision.utils.save_image(
                        attack.denormalize(adv_images_i).cpu(), adv_path
                    )
                    torchvision.utils.save_image(perturbation.cpu().squeeze(), diff_path)

                # Collect data for CSV
                csv_results.append(
                    {
                        "attack_type": "universal",
                        "perturbation_type": perturbation_type,
                        "dataset": self.args.dataset_name,
                        "image_name": base_name,
                        "subdir": subdirs[i],
                        "label": labels[i].item(),
                        "normal_score": f"{orig_scores[i].item():.4f}",
                        "adv_score": f"{adv_score:.4f}",
                        "perturbation_l2": f"{perturbation.norm(2).item():.4f}",
                        "perturbation_max": f"{perturbation.max().item():.4f}",
                        "perturbation_mean": f"{perturbation.mean().item():.4f}",
                        "detectors": ",".join(detectors.keys()),
                        **self.csv_image_paths(result, orig_path, adv_path, diff_path),
                    }
                )

            if self.args.debug and batch_idx >= 2:
                break

            # Clean up GPU memory
            del images
            if real_images is not None:
                del adv_real, real_images
            if fake_images is not None:
                del adv_fake, fake_images
            torch.cuda.empty_cache()

        # Save CSV results
        result.save_csv_results(csv_results)
        results.append(result)

        return results

    def run(self) -> Dict[str, Any]:
        """Execute attack pipeline based on args."""
        if self.args.attack_type in ["ensemble", "diverse", "universal"]:
            transform = get_detector_default_transform(self.args.detector_name)
        else:
            detector, transform = self.initialize_detector()

        _, dataloader = load_dataset(self.args, transform=transform, logger=self.logger)
        config = self.load_attack_config()

        if self.args.attack_type in ["ensemble", "diverse", "universal"]:
            detectors = {}
            detector_names = (
                self.args.ensemble_detectors.split(",")
                if self.args.ensemble_detectors
                else ["drct_clip_sd14", "corvi_sd14", "drct_convb_sd14", "ojha_sd14"]
            )
            detector_names = [
                name.strip()
                for name in detector_names
                if name.strip() != self.args.detector_name
            ]
            if self.args.ensemble_detectors == "all":
                detector_names = [
                    "drct_clip_sd14",
                    "corvi_sd14",
                    "drct_convb_sd14",
                    "ojha_sd14",
                ]
            self.logger.info(f"Using ensemble of detectors: {detector_names}")

            for name in detector_names:
                det = load_detector(
                    name,
                    attack_mode=True,
                    checkpoint_dir=self.args.checkpoint_dir,
                    checkpoint_overrides=self.args.checkpoint,
                )
                det = det.to(self.device)
                det.eval()
                detectors[name] = det
        else:
            detectors = {self.args.detector_name: detector}

        try:
            if self.args.attack_type == "pgd":
                return self.run_pgd_attack(detector, dataloader, config)
            elif self.args.attack_type == "ensemble":
                return self.run_ensemble_attack(detectors, dataloader, config)
            elif self.args.attack_type == "diverse":
                return self.run_diverse_scale_attack(detectors, dataloader, config)
            elif self.args.attack_type == "universal":
                return self.run_universal_attack(detectors, dataloader, config)
            elif self.args.attack_type == "query":
                return self.run_query_efficient_attack(detectors, dataloader, config)
            else:
                raise ValueError(f"Unknown attack type: {self.args.attack_type}")

        except Exception as error:
            self.logger.error(f"Attack failed: {str(error)}", exc_info=True)
            raise


def main() -> None:
    """Entry point for attack script"""
    args = get_args("attack")
    logger = initialize_logger(args)

    try:
        runner = AttackRunner(args, logger)
        results = runner.run()
        logger.info("Attack completed successfully")

        # Convert results to JSON serializable format
        if isinstance(results, list):
            results_dict = [r.to_dict() for r in results]
        elif isinstance(results, dict):
            results_dict = {
                k: [r.to_dict() for r in v] if isinstance(v, list) else v.to_dict()
                for k, v in results.items()
            }
        else:
            results_dict = results.to_dict()

        logger.info(f"Results: {json.dumps(results_dict, indent=2)}")
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
