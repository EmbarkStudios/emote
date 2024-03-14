from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LinearLR

from emote.callbacks.loss import LossCallback
from emote.nn.layers import Conv2dEncoder


def soft_update_from_to(source_params, target_params, tau):
    for target_param, param in zip(target_params, source_params):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def rand_uniform(minval: float, maxval: float, shape: List[int]):
    range = maxval - minval
    rand = torch.rand(shape)
    return range * rand + minval


@torch.jit.script
class ImageAugmentor:
    def __init__(
        self,
        device: torch.device,
        use_fast_augment: bool = True,
        use_noise_aug: bool = True,
        use_per_image_mask_size: bool = False,
        min_mask_relative_size: float = 0.2,  # min size of the mask relative to the size of the image
        max_mask_relative_size: float = 0.4,  # max size of the mask relative to the size of the image
    ):
        self._use_noise_aug = use_noise_aug
        self._use_fast_augment = use_fast_augment
        self._use_per_image_mask_size = use_per_image_mask_size
        self._min_mask_size = min_mask_relative_size
        self._max_mask_size = max_mask_relative_size
        self._device = device

    def __call__(self, image: torch.Tensor):
        with torch.no_grad():
            if self._use_noise_aug:
                image = self._maybe_add_noise(image, noise_std=0.015, noise_prob=0.25)
            if self._use_fast_augment:
                image = self._cutout_per_batch_pos_and_mask_size(image)
            else:
                if self._use_per_image_mask_size:
                    image = self._cutout_per_image_mask_size(image)
                else:
                    image = self._cutout_per_batch_mask_size(image)
        return image

    def _get_mask_indices(
        self,
        image_size_x: int,
        image_size_y: int,
        num_slices: List[int],  # the number of unique index slices to return
    ):
        size = rand_uniform(minval=self._min_mask_size, maxval=self._max_mask_size, shape=[1])[0]
        mask_size: List[int] = [int(image_size_x * size), int(image_size_y * size)]
        start_i = torch.randint(low=0, high=image_size_x - mask_size[0], size=num_slices)
        start_j = torch.randint(low=0, high=image_size_y - mask_size[1], size=num_slices)
        end_i = start_i + mask_size[0]
        end_j = start_j + mask_size[1]
        return start_i, start_j, end_i, end_j

    def _maybe_add_noise(self, image: torch.Tensor, noise_std: float, noise_prob: float):
        prob_sample = rand_uniform(minval=0.0, maxval=1.0, shape=[1])[0]
        # Add noise to the image from a normal distribution.
        if prob_sample < noise_prob:
            image = image + torch.normal(
                mean=0.0, std=noise_std, size=image.shape, device=self._device
            )
        return image

    def _cutout_per_image_mask_size(self, images: torch.Tensor):
        # This is slightly slower than per batch version but in principle it should also be slightly better.
        batch_size, im_x, im_y, _ = images.shape

        for i in range(batch_size):
            start_i, start_j, end_i, end_j = self._get_mask_indices(im_x, im_y, num_slices=[1])
            images[i, start_i:end_i, start_j:end_j, :] = 0
        return images

    def _cutout_per_batch_mask_size(self, images: torch.Tensor):
        batch_size, im_x, im_y, _ = images.shape

        start_i, start_j, end_i, end_j = self._get_mask_indices(im_x, im_y, num_slices=[batch_size])

        for i in range(batch_size):
            images[i, start_i[i] : end_i[i], start_j[i] : end_j[i], :] = 0
        return images

    def _cutout_per_batch_pos_and_mask_size(self, images: torch.Tensor):
        _, im_x, im_y, _ = images.shape
        start_i, start_j, end_i, end_j = self._get_mask_indices(im_x, im_y, num_slices=[1])
        images[:, start_i:end_i, start_j:end_j, :] = 0
        return images


class CurlLoss(LossCallback):
    """Contrastive Unsupervised Representations for Reinforcement Learning
    (CURL).

    paper: https://arxiv.org/abs/2004.04136

    :param encoder_model: (Conv2dEncoder) The image encoder that will be trained using CURL.
    :param target_encoder_model: (Conv2dEncoder) The target image encoder.
    :param device: (torch.device) The device to use for computation.
    :param learning_rate: (float)
    :param learning_rate_start_frac: (float) The start fraction for LR schedule.
    :param learning_rate_end_frac: (float) The end fraction for LR schedule.
    :param learning_rate_steps: (int) The number of step to decay the LR over.
    :param max_grad_norm: (float) The maximum gradient norm, use for gradient clipping.
    :param desired_zdim: (int) The size of the latent. If the projection layer is not used this will
                       default to the encoder output size.
    :param tau: (float) The tau value that is used for updating the target encoder.
    :param use_noise_aug: (bool) Add noise during image augmentation.
    :param temperature: (float) The value used for the temperature scaled cross-entropy calculation.
    :param use_temperature_variant: (bool) Use normalised temperature scaled cross-entropy variant.
    :param use_per_image_mask_size: (bool) Use different mask sizes for every image in the batch.
    :param use_fast_augment: (bool) A gpu compatible image augmentation that uses a fixed cutout
                                    position and size per batch.
    :param use_projection_layer: (bool) Add an additional dense layer to the encoder that projects
                                        to zdim size.
    :param augment_anchor_and_pos: (bool) Augment both the anchor and positive images.
    :param log_images: (bool) Logs the augmented images.
    """

    def __init__(
        self,
        encoder_model: Conv2dEncoder,
        target_encoder_model: Conv2dEncoder,
        device: torch.DeviceObjType,
        learning_rate: float,
        learning_rate_start_frac: float = 1.0,
        learning_rate_end_frac: float = 1.0,
        learning_rate_steps: float = 1,
        max_grad_norm: float = 1.0,
        data_group: str = "default",
        desired_zdim: int = 128,  # This will be ignored if use_projection_layer = False
        tau: float = 0.005,
        use_noise_aug: bool = False,
        temperature: float = 0.1,
        use_temperature_variant: bool = True,
        use_per_image_mask_size: bool = False,
        use_fast_augment: bool = False,
        use_projection_layer: bool = True,
        augment_anchor_and_pos: bool = True,  # disabling this saves some computation and doesn't seem to have any adverse effects.
        log_images: bool = True,
    ):
        self._max_grad_norm = max_grad_norm
        self.data_group = data_group
        self._device = device

        self._use_projection_layer = use_projection_layer
        self._log_images = log_images
        self._use_temperature_variant = use_temperature_variant
        self._augment_anchor_and_pos = augment_anchor_and_pos
        self._tau = tau

        encoder_output_size = encoder_model.get_encoder_output_size()

        if not encoder_model.flatten:
            encoder_output_size = (
                encoder_output_size[0] * encoder_output_size[1] * encoder_output_size[2]
            )
            encoder_model = nn.Sequential(encoder_model, nn.Flatten())

        if not target_encoder_model.flatten:
            target_encoder_model = nn.Sequential(target_encoder_model, nn.Flatten())

        if self._use_projection_layer:
            # Add a layer to reduce the encoder output size to the size of zdim.
            # This differs from the original paper.
            self._zdim = desired_zdim

            # Add projection layer to the encoder.
            encoder_proj_layer = nn.Linear(encoder_output_size, desired_zdim, device=device)
            self._proj_layer_source_vars = encoder_proj_layer.parameters()
            encoder_model = nn.Sequential(encoder_model, encoder_proj_layer, nn.ReLU())

            # Add projection layer to the target encoder.
            target_proj_layer = nn.Linear(encoder_output_size, desired_zdim, device=device)
            self._proj_layer_target_vars = target_proj_layer.parameters()
            target_encoder_model = nn.Sequential(target_encoder_model, target_proj_layer, nn.ReLU())

            # Update the projection layers on the target to match the source
            soft_update_from_to(self._proj_layer_source_vars, self._proj_layer_target_vars, tau=1.0)
        else:
            self._zdim = encoder_output_size

        self._encoder = encoder_model
        self._target_encoder = target_encoder_model

        self._W = torch.tensor(
            torch.rand(size=[desired_zdim, desired_zdim]),
            requires_grad=True,
            device=device,
        )

        if self._use_temperature_variant:
            self._temperature = torch.tensor(
                temperature, requires_grad=False, dtype=torch.float32, device=device
            )

        self._augment = ImageAugmentor(
            use_fast_augment=use_fast_augment,
            use_noise_aug=use_noise_aug,
            use_per_image_mask_size=use_per_image_mask_size,
            device=device,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        lr_schedule = LinearLR(
            optimizer,
            learning_rate_start_frac,
            learning_rate_end_frac,
            learning_rate_steps,
        )
        super().__init__(
            name="curl",
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            network=None,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )

    def parameters(self):
        return list(self._encoder.parameters()) + [self._W]

    def backward(self, observation):
        images = observation["images"]
        image_aug1 = self._augment(images.clone())
        image_aug2 = (
            self._augment(images.clone()) if self._augment_anchor_and_pos else images.clone()
        )

        self.optimizer.zero_grad()
        loss = self._loss(image_aug1, image_aug2)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self._max_grad_norm)
        self.optimizer.step()
        self.lr_schedule.step()

        self.log_scalar(f"loss/{self.name}_lr", self.lr_schedule.get_last_lr()[0])
        self.log_scalar(f"loss/{self.name}_loss", loss)
        self.log_scalar(f"loss/{self.name}_gradient_norm", grad_norm)

        if self._log_images:
            self.log_image("augmentations/base_allch", images[0, :, :, :])
            self.log_image("augmentations/image1_allch", image_aug1[0, :, :, :])
            self.log_image("augmentations/image2_allch", image_aug2[0, :, :, :])

    @torch.jit.export
    def _loss(self, image1: torch.Tensor, image2: torch.Tensor):
        batch_size = image1.shape[0]

        # ENCODE
        z_a = self._encoder(image1)
        with torch.no_grad():
            z_pos = self._target_encoder(image2)

        # PROJECTION
        Wz = self._W @ z_pos.T  # (z,B)

        # LOGITS
        logits = z_a @ Wz  # (B,B)

        if self._use_temperature_variant:
            # Use normalised temperature scaled cross-entropy. This differs from the orig
            # CURL paper but it seems to give better results. This technique is also used
            # in SimCLR v2.
            logits = logits / self._temperature
        else:
            # remove max for numerical stability
            logits = logits - torch.amax(logits, dim=1)

        # LOSS
        # One neat trick!: Diags are positive examples, off-diag are negative examples!
        labels = F.one_hot(torch.arange(batch_size, device=self._device), batch_size).float()
        loss = (-labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return torch.mean(loss)

    def end_batch(self):
        if self._use_projection_layer:
            soft_update_from_to(
                self._proj_layer_source_vars,
                self._proj_layer_target_vars,
                tau=self._tau,
            )
