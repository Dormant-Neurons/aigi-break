import logging

import torch
import torch.nn


class BaseAttack:
    """Base class for all attacks"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.logger = logging.getLogger(self.__class__.__name__)

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def get_fakeness_score(self, model, x):
        """Get fakeness score from model output"""
        output = model(x)
        if output.shape[-1] == 2:  # Two logits [real, fake]
            return torch.softmax(output, dim=1)[:, 1]
        return torch.sigmoid(output.squeeze())

    def compute_loss(self, score, target_is_fake):
        """
        Compute loss based on target prediction.
        If target_is_fake=True, we want model to predict fake (score→1)
        If target_is_fake=False, we want model to predict real (score→0)

        Args:
            score: Model prediction scores (0=real, 1=fake)
            target_is_fake: Target prediction (True=fake/1, False=real/0)
        Returns:
            Loss tensor with same batch dimension as input
        """
        target = self._target_tensor(target_is_fake, score)
        return torch.where(target, -score, score)

    @staticmethod
    def _target_tensor(target_is_fake, reference):
        """Return target labels as a bool tensor broadcastable to reference."""
        if isinstance(target_is_fake, torch.Tensor):
            target = target_is_fake.to(device=reference.device, dtype=torch.bool)
        else:
            target = torch.as_tensor(
                target_is_fake, device=reference.device, dtype=torch.bool
            )
        if target.ndim == 0 or target.numel() == 1:
            target = target.reshape(()).expand_as(reference)
        return target.reshape(reference.shape)

    def log_progress(self, step, total_steps, loss, score, **kwargs):
        """Log attack progress"""
        extra = " ".join([f"{k}: {v:.4f}" for k, v in kwargs.items()])
        self.logger.info(
            f"Step [{step+1}/{total_steps}] Loss: {loss:.4f} "
            f"Score: {score:.4f} {extra}"
        )


class PGDAttack(BaseAttack):
    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(self, images, target_is_fake):
        """
        Perform PGD attack
        Args:
            images: Input images to attack
            target_is_fake: Target prediction (True=fake, False=real)
        """
        images = images.clone().detach()
        images_denorm = self.denormalize(images)
        adv_images = images_denorm.clone().detach()

        best_loss = float("nan")
        best_adv = adv_images.clone()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for step in range(self.steps):
            adv_images.requires_grad = True
            score = self.get_fakeness_score(self.model, self.normalize(adv_images))
            loss = self.compute_loss(score, target_is_fake)
            loss_mean = loss.mean()

            self.log_progress(
                step,
                self.steps,
                loss_mean.item(),
                score.mean().item(),
                best_loss=best_loss,
            )

            grad = torch.autograd.grad(loss_mean, adv_images)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()

            delta = torch.clamp(adv_images - images_denorm, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images_denorm + delta, min=0, max=1).detach()
            best_loss = loss_mean.item()
            best_adv = adv_images.clone().detach()

        return self.normalize(best_adv)


class EnsembleAttack(BaseAttack):
    def __init__(
        self,
        models,
        model_transforms,
        weights=None,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        random_start=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models = models if isinstance(models, list) else [models]
        self.model_transforms = (
            model_transforms
            if isinstance(model_transforms, list)
            else [model_transforms]
        )

        if len(self.models) != len(self.model_transforms):
            raise ValueError("Number of models must match number of transforms")

        # Initialize weights if not provided
        if weights is None:
            weights = [1.0] * len(self.models)
        weights = torch.tensor(weights)
        self.weights = weights / weights.sum()  # Normalize weights

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def get_ensemble_score(self, x):
        """Get weighted average of fakeness scores from all models"""
        ensemble_scores = []
        for model, transform, weight in zip(
            self.models, self.model_transforms, self.weights
        ):
            # Apply model-specific transform if provided
            if transform is not None:
                x_transformed = transform(self.denormalize(x))
            else:
                x_transformed = x
            score = self.get_fakeness_score(model, x_transformed)
            ensemble_scores.append(weight.to(x.device) * score)
        return sum(ensemble_scores)

    def attack(self, images, target_is_fake):
        """
        Perform ensemble PGD attack
        Args:
            images: Input images to attack
            target_is_fake: Target prediction (True=fake, False=real)
        """
        images = images.clone().detach()
        images_denorm = self.denormalize(images)
        adv_images = images_denorm.clone().detach()

        best_loss = float("nan")
        best_adv = adv_images.clone()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for step in range(self.steps):
            adv_images.requires_grad = True

            # Get ensemble score
            score = self.get_ensemble_score(self.normalize(adv_images))
            loss = self.compute_loss(score, target_is_fake)
            loss_mean = loss.mean()

            self.log_progress(
                step,
                self.steps,
                loss_mean.item(),
                score.mean().item(),
                best_loss=best_loss,
            )

            # Compute gradient and update
            grad = torch.autograd.grad(loss_mean, adv_images)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()

            # Project perturbation onto epsilon ball
            delta = torch.clamp(adv_images - images_denorm, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images_denorm + delta, min=0, max=1).detach()
            best_loss = loss_mean.item()
            best_adv = adv_images.clone().detach()

        return self.normalize(best_adv)


class DiverseScaleAttack(BaseAttack):
    def __init__(
        self,
        models,
        model_transforms,
        weights=None,
        eps=8 / 255,
        alpha=2 / 255,
        steps=50,
        random_start=True,
        momentum=0.9,
        resize_prob=0.8,
        div_prob=0.8,
        scale_min=0.8,
        scale_max=1.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models = models if isinstance(models, list) else [models]
        self.model_transforms = (
            model_transforms
            if isinstance(model_transforms, list)
            else [model_transforms]
        )

        if len(self.models) != len(self.model_transforms):
            raise ValueError("Number of models must match number of transforms")

        if weights is None:
            weights = [1.0] * len(self.models)
        weights = torch.tensor(weights)
        self.weights = weights / weights.sum()  # Normalize weights

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.momentum = momentum
        self.resize_prob = resize_prob
        self.div_prob = div_prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.device = next(self.models[0].parameters()).device

    def input_diversity(self, x):
        """Apply input diversity with gradient tracking"""
        if torch.rand(1) < self.resize_prob:
            orig_size = x.shape[-1]
            scale_factor = self.scale_min + torch.rand(1) * (
                self.scale_max - self.scale_min
            )
            new_size = int(orig_size * scale_factor)

            # Random resize with gradient tracking
            x_resized = torch.nn.functional.interpolate(
                x, size=new_size, mode="bilinear", align_corners=False
            )

            if new_size < orig_size:
                pad_size = orig_size - new_size
                pad_left = torch.randint(0, pad_size + 1, (1,)).item()
                pad_right = pad_size - pad_left
                pad_top = torch.randint(0, pad_size + 1, (1,)).item()
                pad_bottom = pad_size - pad_top
                x_resized = torch.nn.functional.pad(
                    x_resized, (pad_left, pad_right, pad_top, pad_bottom)
                )
            else:
                crop_start = (new_size - orig_size) // 2
                x_resized = x_resized[
                    :,
                    :,
                    crop_start : crop_start + orig_size,
                    crop_start : crop_start + orig_size,
                ]

            if torch.rand(1) < self.div_prob:
                return x_resized
        return x

    def get_ensemble_score(self, x):
        """Get weighted average of fakeness scores from all models with diverse scaling"""
        ensemble_scores = []
        diverse_x = self.input_diversity(x)

        for model, transform, weight in zip(
            self.models, self.model_transforms, self.weights
        ):
            # Apply model-specific transform if provided
            if transform is not None:
                x_transformed = transform(self.denormalize(diverse_x))
            else:
                x_transformed = diverse_x
            score = self.get_fakeness_score(model, x_transformed)
            ensemble_scores.append(weight.to(x.device) * score)
        return sum(ensemble_scores)

    def attack(self, images, target_is_fake):
        images = images.clone().detach().to(self.device)
        images_denorm = self.denormalize(images)
        adv_images = images_denorm.clone().detach()
        momentum_grad = torch.zeros_like(images_denorm)
        best_adv = adv_images.clone()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for step in range(self.steps):
            adv_images.requires_grad_(True)

            # Compute ensemble score with diversity
            score = self.get_ensemble_score(self.normalize(adv_images))
            loss = self.compute_loss(score, target_is_fake)
            loss_mean = loss.mean()

            # Compute gradient with proper backward pass
            loss_mean.backward()

            with torch.no_grad():
                grad = adv_images.grad
                if grad is None:
                    raise RuntimeError(
                        "No gradient available. Check if all operations maintain gradient flow."
                    )

                # Update momentum and adversarial examples
                momentum_grad = self.momentum * momentum_grad + grad / (
                    torch.norm(grad, p=1) + 1e-10
                )
                adv_images = adv_images.detach() - self.alpha * momentum_grad.sign()
                delta = torch.clamp(
                    adv_images - images_denorm, min=-self.eps, max=self.eps
                )
                adv_images = torch.clamp(images_denorm + delta, min=0, max=1)

                # Update best adversarial example
                best_adv = adv_images.clone()
                self.log_progress(
                    step, self.steps, loss_mean.item(), score.mean().item()
                )

            # Clear gradients for next step
            adv_images.grad = None

        return self.normalize(best_adv)


class UniversalAttack(BaseAttack):
    def __init__(
        self,
        models,
        model_transforms=None,
        weights=None,
        eps=8 / 255,
        alpha=1 / 255,
        steps=50,
        momentum=0.9,
        batch_size=8,
        universal_weight=0.8,
        dual_perturbation=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models = models if isinstance(models, list) else [models]
        self.model_transforms = (
            model_transforms
            if isinstance(model_transforms, list)
            else [None] * len(self.models)
        )

        if len(self.models) != len(self.model_transforms):
            raise ValueError("Number of models must match number of transforms")

        if weights is None:
            weights = [1.0] * len(self.models)
        weights = torch.tensor(weights)
        self.weights = weights / weights.sum()

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.batch_size = batch_size
        self.universal_weight = universal_weight
        self.dual_perturbation = dual_perturbation
        # Initialize separate deltas for fake->real and real->fake
        self.delta_fake_to_real = (
            None  # For making fake images look real (target_is_fake=False)
        )
        self.delta_real_to_fake = (
            None  # For making real images look fake (target_is_fake=True)
        )

    def get_ensemble_score(self, x):
        """Get weighted average of fakeness scores from all models"""
        ensemble_scores = []
        for model, transform, weight in zip(
            self.models, self.model_transforms, self.weights
        ):
            # Apply model-specific transform if provided
            if transform is not None:
                x_transformed = transform(self.denormalize(x))
            else:
                x_transformed = x
            score = self.get_fakeness_score(model, x_transformed)
            ensemble_scores.append(weight.to(x.device) * score)
        return sum(ensemble_scores)

    def train_step(self, images, universal_delta, momentum_grad, target_is_fake):
        """Single training step for universal perturbation with target direction"""
        images_denorm = self.denormalize(images)
        batch_size = images.shape[0]

        # Add universal perturbation
        adv_images = torch.clamp(
            images_denorm + universal_delta.expand(batch_size, -1, -1, -1), min=0, max=1
        )
        adv_images.requires_grad = True

        # Get ensemble score and compute loss
        score = self.get_ensemble_score(self.normalize(adv_images))

        # Calculate loss based on target direction
        # For target_is_fake=True (real→fake): we want to maximize the fakeness score
        # For target_is_fake=False (fake→real): we want to minimize the fakeness score
        loss = self.compute_loss(score, target_is_fake).mean()

        grad = torch.autograd.grad(loss, adv_images)[0]
        grad = grad.mean(dim=0, keepdim=True)  # Average across batch

        # Update momentum
        momentum_grad = self.momentum * momentum_grad + grad / torch.norm(grad, p=1)

        return momentum_grad, loss.item(), score.mean().item()

    def generate_perturbation(self, dataloader, target_is_fake=None):
        """
        Generate universal adversarial perturbation

        Args:
            dataloader: Data loader with images
            target_is_fake: If True, generate perturbation to make images look fake
                           If False, generate perturbation to make images look real
                           If None and self.dual_perturbation=True, generate both
        """
        # First, verify if we need to generate both perturbations or just one
        if target_is_fake is None and self.dual_perturbation:
            self.logger.info(
                "Generating both fake→real and real→fake universal perturbations..."
            )

            # Split dataset by labels
            fake_images = []
            real_images = []

            for batch in dataloader:
                # Handle different return types from dataloader
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    continue  # Skip if no labels available

                # Sort images by label
                for i, label in enumerate(labels):
                    if label == 1:  # Fake
                        fake_images.append(images[i : i + 1])
                    else:  # Real
                        real_images.append(images[i : i + 1])

            # Convert to batches if enough data
            if fake_images:
                # Generate fake→real perturbation using fake images (target_is_fake=False)
                self.logger.info("\nGenerating fake→real perturbation...")
                self._generate_single_perturbation(dataloader, False)

            if real_images:
                # Generate real→fake perturbation using real images (target_is_fake=True)
                self.logger.info("\nGenerating real→fake perturbation...")
                self._generate_single_perturbation(dataloader, True)

            return {
                "fake_to_real": self.delta_fake_to_real,
                "real_to_fake": self.delta_real_to_fake,
            }
        else:
            # Just generate a single perturbation based on target direction
            return self._generate_single_perturbation(
                dataloader, target_is_fake or False
            )

    def _generate_single_perturbation(self, dataloader, target_is_fake):
        """Generate a single universal perturbation based on target direction"""
        device = next(self.models[0].parameters()).device
        direction = "real→fake" if target_is_fake else "fake→real"
        self.logger.info(f"\nStarting {direction} Universal Perturbation Generation...")
        self.logger.info(
            f"Parameters: eps={self.eps:.4f}, alpha={self.alpha:.4f}, steps={self.steps}"
        )

        # Initialize universal perturbation
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (tuple, list)):
            sample_batch = sample_batch[0]
        delta = torch.zeros(1, *sample_batch.shape[1:]).to(device)
        momentum_grad = torch.zeros_like(delta)

        best_loss = float("inf")
        no_improve_count = 0

        for step in range(self.steps):
            total_loss = 0
            total_score = 0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (tuple, list)):
                    images, labels = batch[0], batch[1]
                    # Only process images that match our target direction
                    if target_is_fake and all(label != 0 for label in labels):
                        continue  # Skip batch if we're targeting real→fake but no real images
                    if not target_is_fake and all(label != 1 for label in labels):
                        continue  # Skip batch if we're targeting fake→real but no fake images
                else:
                    images = batch

                images = images.to(device)

                momentum_grad, loss, score = self.train_step(
                    images, delta, momentum_grad, target_is_fake
                )

                # Update universal perturbation
                delta = delta.detach() - self.alpha * momentum_grad.sign()
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)

                total_loss += loss
                total_score += score
                num_batches += 1

                if num_batches >= 10 and self.steps > 5:
                    # Early stop batch processing for faster development
                    break

            if num_batches == 0:
                self.logger.warning(
                    f"No suitable images found for {direction} perturbation in this epoch"
                )
                continue

            # Calculate averages
            avg_loss = total_loss / num_batches
            avg_score = total_score / num_batches

            # Track best loss and improvement
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            pert_magnitude = torch.norm(delta).item()
            self.logger.info(
                f"{direction} - Step [{step+1}/{self.steps}] "
                f"Loss: {avg_loss:.4f} "
                f"Score: {avg_score:.4f} "
                f"Perturbation Magnitude: {pert_magnitude:.4f} "
                f"No Improvement Count: {no_improve_count}"
            )

        # Store the perturbation in the appropriate attribute
        if target_is_fake:
            self.delta_real_to_fake = delta
        else:
            self.delta_fake_to_real = delta

        return delta

    def attack(self, images, target_is_fake=None):
        """
        Apply universal perturbation to images

        Args:
            images: Input images to perturb
            target_is_fake: If True, make images look fake, if False, make images look real
                            If None, use label to determine direction (real→fake for real images, fake→real for fake)
        """
        if self.dual_perturbation:
            if self.delta_fake_to_real is None or self.delta_real_to_fake is None:
                raise ValueError("Must call generate_perturbation first!")
        else:
            if self.delta_fake_to_real is None and self.delta_real_to_fake is None:
                raise ValueError("Must call generate_perturbation first!")

        images_denorm = self.denormalize(images)
        batch_size = images.shape[0]

        if target_is_fake is None and self.dual_perturbation:
            # Use batch inference - apply different perturbation to each image based on its label
            # Assuming images come with labels in a tuple (images, labels, ...)
            if not hasattr(images, "labels"):
                self.logger.warning(
                    "No labels provided for dual perturbation. Defaulting to fake→real."
                )
                # Apply fake→real perturbation to all images if no labels
                delta = self.delta_fake_to_real
            else:
                # Create a batch of mixed perturbations
                delta = torch.zeros_like(images_denorm)
                for i, label in enumerate(images.labels):
                    if label == 0:  # Real image, apply real→fake
                        delta[i] = self.delta_real_to_fake
                    else:  # Fake image, apply fake→real
                        delta[i] = self.delta_fake_to_real
        else:
            # Use single direction for all images in batch
            if target_is_fake:
                if self.delta_real_to_fake is None:
                    raise ValueError("Real→fake perturbation not generated yet!")
                delta = self.delta_real_to_fake.expand(batch_size, -1, -1, -1)
            else:
                if self.delta_fake_to_real is None:
                    raise ValueError("Fake→real perturbation not generated yet!")
                delta = self.delta_fake_to_real.expand(batch_size, -1, -1, -1)

        # Apply universal perturbation
        adv_images = torch.clamp(images_denorm + delta, min=0, max=1)

        return self.normalize(adv_images)


class QueryEfficientAttack(BaseAttack):
    def __init__(
        self,
        model,
        eps=8 / 255,
        max_queries=100,
        samples_per_draw=10,
        noise_scale=0.005,
        min_step_factor=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.eps = eps
        self.max_queries = max_queries
        self.samples_per_draw = samples_per_draw
        self.noise_scale = noise_scale  # Tunable noise scale factor
        self.min_step = self.eps * min_step_factor  # Minimum step size
        self.query_count = 0  # Initialize query count in __init__

    def estimate_gradient(self, x, target_is_fake):
        """Estimate gradient using random sampling with extremely efficient memory usage"""
        batch_size = x.shape[0]

        # Get base score for the current image
        base_score = self.get_fakeness_score(self.model, self.normalize(x))
        self.query_count += batch_size  # Count base evaluation

        # Initialize gradient accumulator (don't create full tensor right away)
        grad = None

        target = self._target_tensor(target_is_fake, base_score)
        direction = torch.where(
            target,
            torch.ones_like(base_score),
            -torch.ones_like(base_score),
        ).view(batch_size, 1, 1, 1)

        # Process one noise sample at a time to minimize memory usage
        for i in range(self.samples_per_draw):
            # Generate just one noise sample at a time
            with torch.no_grad():  # Use no_grad to avoid tracking history
                # Generate and normalize a single noise sample
                noise = torch.randn(batch_size, *x.shape[1:], device=x.device)
                noise_norm = torch.sqrt(
                    torch.sum(noise**2, dim=(1, 2, 3), keepdim=True)
                )
                noise = noise / (noise_norm + 1e-10)

                # Apply noise to image
                perturbed = torch.clamp(x + noise * self.eps * self.noise_scale, 0, 1)

                # Get score for perturbed image
                perturbed_score = self.get_fakeness_score(
                    self.model, self.normalize(perturbed)
                )
                self.query_count += batch_size

                # Calculate score difference
                score_diff = perturbed_score - base_score

                # Update gradient accumulator
                weighted_noise = (
                    score_diff.view(batch_size, 1, 1, 1) * noise * direction
                )

                if grad is None:
                    grad = weighted_noise
                else:
                    grad += weighted_noise

                # Free memory immediately
                del (
                    noise,
                    noise_norm,
                    perturbed,
                    perturbed_score,
                    score_diff,
                    weighted_noise,
                )

            # Force garbage collection after each iteration
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Optional: add a small delay to ensure memory is properly freed
            import time

            time.sleep(0.01)

        return grad / self.samples_per_draw

    def attack(self, images, target_is_fake):
        """
        Perform query-efficient black-box attack
        Args:
            images: Input images to attack
            target_is_fake: Whether images are fake (True) or real (False)
        Returns:
            Adversarial examples
        """
        self.query_count = 0  # Reset query count at start of attack
        images = images.clone().detach()
        images_denorm = self.denormalize(images)
        adv_images = images_denorm.clone()
        best_adv = adv_images.clone()

        # Initial score
        curr_score = self.get_fakeness_score(self.model, images)
        self.query_count += images.shape[0]  # Count initial query
        best_score = curr_score.clone()

        target = self._target_tensor(target_is_fake, curr_score)
        best_objective = torch.where(target, best_score, -best_score)

        step_size = self.eps * 0.8
        no_improvement_count = 0

        while self.query_count < self.max_queries:
            # Estimate gradient
            grad = self.estimate_gradient(adv_images, target_is_fake)

            # Update adversarial examples - add gradient because estimate_gradient returns in correct direction
            adv_images = adv_images + step_size * torch.sign(grad)
            delta = torch.clamp(adv_images - images_denorm, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images_denorm + delta, min=0, max=1).detach()

            # Evaluate new score
            curr_score = self.get_fakeness_score(self.model, self.normalize(adv_images))
            self.query_count += images.shape[0]  # Count evaluation query

            # Update best adversarial example if score improved
            curr_objective = torch.where(target, curr_score, -curr_score)
            if curr_objective.mean() > best_objective.mean():
                best_adv = adv_images.clone()
                best_score = curr_score.clone()
                best_objective = curr_objective.clone()
                no_improvement_count = 0
                self.logger.info(f"Improvement found! Score: {curr_score.mean():.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    # Reduce step size if no improvement for several iterations
                    new_step_size = step_size * 0.5
                    if new_step_size >= self.min_step:  # Check against min_step
                        step_size = new_step_size
                        self.logger.info(f"Reducing step size to {step_size:.6f}")
                    else:
                        self.logger.info(
                            f"Step size reached minimum {self.min_step:.6f}. Stopping reduction."
                        )
                    no_improvement_count = 0  # Reset count after adjusting step size

            self.logger.info(
                f"Queries: {self.query_count}, Current Score: {curr_score.mean():.4f}, "
                f"Best Score: {best_score.mean():.4f}, Step size: {step_size:.6f}"
            )

            # Check if we're about to exceed query limit
            if (
                self.query_count + (self.samples_per_draw + 2) * images.shape[0]
                > self.max_queries
            ):
                self.logger.info("Stopping early to not exceed query limit")
                break

        self.logger.info(
            f"Attack completed with {self.query_count} queries. Final score: {best_score.mean():.4f}"
        )
        return self.normalize(best_adv)
