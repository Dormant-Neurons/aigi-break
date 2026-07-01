import collections
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

# External dependencies
from external.corvi.test_code.get_method_here import def_model, get_method_here
from external.drct.network.models import get_models as get_drct_base_models
from external.ojha.models import get_model as get_ojha_base_model

# Configure logging
logger = logging.getLogger(__name__)
MODEL_BASE_PATH = Path("data/models")
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("ADVDF_CHECKPOINT_DIR", "data/checkpoints")
)
CheckpointOverrides = Union[Mapping[str, str], Sequence[str]]


@dataclass
class DetectorConfig:
    """Configuration container for model loading parameters"""

    name: str
    type: str
    path: Optional[str] = None
    checkpoint: Optional[str] = None
    legacy_path: Optional[str] = None
    model_params: Tuple = ()
    arch: Optional[str] = None
    target_layer: Optional[str] = None
    base_model: Optional[str] = None
    needs_wrapper: bool = True
    experimental: bool = False
    deprecated: bool = False


DETECTOR_CONFIG: Dict[str, Dict] = {
    "corvi_sd14": {
        "type": "corvi",
        "checkpoint": "corvi-sd14.pth",
        "legacy_path": str(
            MODEL_BASE_PATH / "corvi/weights/Grag2021_sd14/model_epoch_best.pth"
        ),
        "arch": "res50stride1",
    },
    "ojha_sd14": {
        "type": "ojha",
        "checkpoint": "ojha-sd14.pth",
        "legacy_path": str(MODEL_BASE_PATH / "ojha/ojha_sd14.pth"),
        "target_layer": "fc",
    },
    "drct_clip_sd14": {
        "type": "drct",
        "checkpoint": "drct-sd14-clip-vit-l14.pth",
        "legacy_path": str(
            MODEL_BASE_PATH
            / "drct/authors_weights/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth"
        ),
        "base_model": "clip",
        "needs_wrapper": False,
    },
    "drct_convb_sd14": {
        "type": "drct",
        "checkpoint": "drct-sd14-convnext-base.pth",
        "legacy_path": str(
            MODEL_BASE_PATH
            / "drct/authors_weights/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"
        ),
        "base_model": "convb",
        "needs_wrapper": False,
    },
    # Robust ojha_sd14 models
    "ojha_sd14_r4": {
        "type": "ojha",
        "checkpoint": "ojha-sd14-robust-eps4.pth",
        "legacy_path": str(MODEL_BASE_PATH / "ojha/robust_clip_eps4_backbone.pth"),
        "experimental": True,
    },
    "ojha_sd14_r2": {
        "type": "ojha",
        "checkpoint": "ojha-sd14-robust-eps2.pth",
        "legacy_path": str(MODEL_BASE_PATH / "ojha/robust_clip_eps2_backbone.pth"),
        "experimental": True,
    },
}


class AttackModeWrapper(torch.nn.Module):
    """Base class for attack mode compatible models"""

    def __init__(self, model: torch.nn.Module, attack_mode: bool):
        super().__init__()
        self.model = model
        self.attack_mode = attack_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OjhaWrapper(AttackModeWrapper):
    """Wrapper for Ojha-style models with attack mode support"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if self.attack_mode:
            return torch.stack([-logits, logits], dim=1).reshape(-1, 2)
        return logits.reshape(-1)


class CorviWrapper(AttackModeWrapper):
    """Wrapper for Corvi-style models with spatial averaging"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x).mean(dim=(1, 2, 3))
        if self.attack_mode:
            return torch.stack([-logits, logits], dim=1)
        return logits


def _parse_checkpoint_overrides(
    checkpoint_overrides: Optional[CheckpointOverrides],
) -> Dict[str, str]:
    if not checkpoint_overrides:
        return {}

    if isinstance(checkpoint_overrides, Mapping):
        overrides = dict(checkpoint_overrides)
    else:
        overrides = {}
        for item in checkpoint_overrides:
            if "=" not in item:
                raise ValueError(
                    "--checkpoint must use DETECTOR=PATH, for example "
                    "--checkpoint ojha_sd14=data/checkpoints/ojha-sd14.pth"
                )
            detector_name, path = item.split("=", 1)
            overrides[detector_name.strip()] = path.strip()

    unknown = sorted(set(overrides) - set(DETECTOR_CONFIG))
    if unknown:
        raise ValueError(f"Unknown checkpoint override(s): {', '.join(unknown)}")

    return overrides


def _resolve_checkpoint_path(
    config: DetectorConfig,
    checkpoint_dir: Optional[str | Path],
    checkpoint_overrides: Optional[CheckpointOverrides],
) -> Optional[str]:
    overrides = _parse_checkpoint_overrides(checkpoint_overrides)
    if config.name in overrides:
        return overrides[config.name]

    if config.path:
        return config.path

    if not config.checkpoint:
        return None

    explicit_dir = checkpoint_dir is not None or "ADVDF_CHECKPOINT_DIR" in os.environ
    base_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    )
    checkpoint_path = base_dir / config.checkpoint

    if checkpoint_path.exists() or explicit_dir or not config.legacy_path:
        return str(checkpoint_path)

    legacy_path = Path(config.legacy_path)
    if legacy_path.exists():
        return str(legacy_path)

    return str(checkpoint_path)


def _load_state_dict(
    filepath: str, target_layer: Optional[str] = None
) -> collections.OrderedDict:
    """Load state dict with FC layer handling"""
    try:
        state = torch.load(filepath, map_location="cpu")
        model_state = state.get("model", state)

        if target_layer:
            # Extract FC layer components
            return collections.OrderedDict(
                [
                    ("weight", model_state[f"{target_layer}.weight"]),
                    ("bias", model_state[f"{target_layer}.bias"]),
                ]
            )

        return model_state

    except KeyError as ke:
        logger.error(f"Missing key {ke} in {filepath}")
        raise
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise RuntimeError(f"Failed loading {filepath}") from e


def _create_ojha_model(config: DetectorConfig, attack_mode: bool) -> torch.nn.Module:
    """Factory function for Ojha model variants"""
    try:
        if not config.path:
            raise ValueError("Model path must be specified")
        model = get_ojha_base_model("CLIP:ViT-L/14")
        state = _load_state_dict(config.path, config.target_layer)

        if config.target_layer:
            getattr(model, config.target_layer).load_state_dict(state)
        else:
            model.load_state_dict(state)

        if config.deprecated:
            logger.warning(f"Using deprecated model: {config.name}")

        return OjhaWrapper(model, attack_mode) if config.needs_wrapper else model

    except Exception as e:
        logger.error(f"Ojha model creation failed: {e}")
        raise


def _create_corvi_model(config: DetectorConfig, attack_mode: bool) -> torch.nn.Module:
    """Factory function for Corvi model variants"""
    try:
        if config.path:
            model = def_model(config.arch or "res50stride1", config.path).cuda().eval()
        else:
            model_params = get_method_here(*config.model_params)
            model = def_model(model_params[2], model_params[1]).cuda().eval()
        return CorviWrapper(model, attack_mode) if config.needs_wrapper else model
    except Exception as e:
        logger.error(f"Corvi model creation failed: {e}")
        raise


def _create_drct_model(config: DetectorConfig, attack_mode: bool) -> torch.nn.Module:
    """Factory function for DRCT models"""
    try:
        base_model = get_drct_base_models(
            model_name="clip-ViT-L-14"
            if config.base_model == "clip"
            else "convnext_base_in22k",
            embedding_size=1024,
            pretrained=False,
            num_classes=2,
            in_channel=3,
            freeze_extractor=False,
        )

        if config.path is None:
            raise ValueError("Model path must be specified")
        state = _load_state_dict(config.path)
        base_model.load_state_dict(state)
        return (
            OjhaWrapper(base_model.eval(), attack_mode)
            if config.needs_wrapper
            else base_model.eval()
        )

    except Exception as e:
        logger.error(f"DRCT model creation failed: {e}")
        raise


def load_detector(
    name: str,
    attack_mode: bool = False,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_overrides: Optional[CheckpointOverrides] = None,
) -> torch.nn.Module:
    """Main entry point for model loading"""
    if name not in DETECTOR_CONFIG:
        raise ValueError(
            f"Unknown detector: {name}. Available: {list(DETECTOR_CONFIG.keys())}"
        )

    config = DetectorConfig(name=name, **DETECTOR_CONFIG[name])
    config.path = _resolve_checkpoint_path(config, checkpoint_dir, checkpoint_overrides)

    if config.experimental:
        logger.warning(f"Loading experimental model: {name}")

    try:
        factory_map = {
            "ojha": _create_ojha_model,
            "corvi": _create_corvi_model,
            "drct": _create_drct_model,
        }
        return factory_map[config.type](config, attack_mode)
    except KeyError:
        raise ValueError(f"Unsupported model type: {config.type}")
