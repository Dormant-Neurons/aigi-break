import io

import numpy as np
from PIL import Image
from torchvision.transforms import v2

corvi_transform_params = {
    "resize": (256, 256),
    "crop": 200,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

ojha_transform_params = {
    "resize": (256, 256),
    "crop": 224,
    "mean": [0.481, 0.457, 0.408],
    "std": [0.268, 0.261, 0.275],
}

drct_transform_params = {
    "resize": (256, 256),
    "crop": 224,
    "mean": [0.481, 0.457, 0.408],
    "std": [0.268, 0.261, 0.275],
}

detector_transform_params = {
    "corvi": corvi_transform_params,
    "ojha": ojha_transform_params,
    "drct": drct_transform_params,
}


def get_detector_default_transform(
    detector_name,
    normalize=True,
    include_to_tensor=True,
):
    """Get default transforms for a detector.

    Args:
        detector_name (str): Name of the detector
        normalize (bool): Whether to apply normalization
        include_to_tensor (bool): Whether to include ToTensor transform
    """

    # Find matching detector config
    detector_config = None
    for key in detector_transform_params:
        if key in detector_name:
            detector_config = detector_transform_params[key]
            break

    if not detector_config:
        raise ValueError(f"Detector {detector_name} not found in transform parameters")

    transforms = []

    if detector_config["resize"] is not None:
        transforms.append(v2.Resize(detector_config["resize"]))

    if detector_config["crop"] is not None:
        transforms.append(
            v2.CenterCrop((detector_config["crop"], detector_config["crop"]))
        )

    if include_to_tensor:
        transforms.append(v2.ToTensor())

    if normalize:
        transforms.append(
            v2.Normalize(mean=detector_config["mean"], std=detector_config["std"])
        )

    return v2.Compose(transforms)


class JPEGCompression:
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        img = Image.open(buffer)
        return img


class AddNoise:
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1) * 255
        return Image.fromarray(np_img.astype(np.uint8))


class ResizeByFraction:
    """Resize an image by a specified fraction while maintaining aspect ratio."""

    def __init__(self, fraction=0.5):
        """
        Args:
            fraction (float): The fraction to scale the image dimensions by.
                            e.g., 0.5 for half size, 0.75 for 3/4 size
        """
        self.fraction = fraction

    def __call__(self, img):
        w, h = img.size
        new_w, new_h = (
            int(w * self.fraction),
            int(h * self.fraction),
        )  # Scale by fraction
        return v2.Resize((new_h, new_w), antialias=True)(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction})"
