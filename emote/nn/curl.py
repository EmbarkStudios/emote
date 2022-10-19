import torch
import torch.nn as nn
import torch.nn.functional as F

from emote.callbacks import LossCallback


def soft_update_from_to(source_params, target_params, tau):
    for target_param, param in zip(target_params, source_params):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def rand_uniform(minval, maxval, shape):
    range = maxval - minval
    rand = torch.rand(shape)
    return range * rand + minval


class CurlLoss(LossCallback):
    """
    Contrastive Unsupervised Representations for Reinforcement Learning (CURL).

    paper: https://arxiv.org/abs/2004.04136

    :param encoder_model: (keras model) The image encoder that will be trained using CURL.
                                        The encoder should have 1 flattened output.
    :param target_encoder_model: (keras model) The target image encoder.
                                               The encoder should have 1 flattened output.
    :param optim: The optimiser that will be using for training.
    :param max_grad_norm: (float) The maximum gradient norm, use for gradient clipping.
    :param lr_shedule: The learning rate schedule.
    :param zdim: (int) The size of the latent. If the projection layer is not used this will
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
    :param name: (str) The callback name.
    """

    def __init__(
        self,
        encoder_model,
        target_encoder_model,
        encoder_output_size: int,
        opt,
        max_grad_norm: float = 1.0,
        data_group: str = "default",
        lr_schedule=None,
        zdim: int = 128,
        tau: float = 0.005,
        use_noise_aug: bool = False,
        temperature: float = 0.1,
        use_temperature_variant: bool = True,
        use_per_image_mask_size: bool = False,
        use_fast_augment: bool = False,
        use_projection_layer: bool = True,
        augment_anchor_and_pos: bool = True,
        log_images: bool = False,
    ):
        super().__init__(
            name="curl",
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=None,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self._max_grad_norm = max_grad_norm
        self.data_group = data_group

        self._use_projection_layer = use_projection_layer
        self._log_images = log_images
        self._use_fast_augment = use_fast_augment
        self._use_noise_aug = use_noise_aug
        self._use_temperature_variant = use_temperature_variant
        self._use_per_image_mask_size = use_per_image_mask_size
        self._augment_anchor_and_pos = augment_anchor_and_pos
        self._tau = tau

        if self._use_projection_layer:
            # Add a layer to reduce the encoder output size to the size of zdim.
            # This differs from the original paper.
            self._zdim = zdim

            encoder_proj_layer = nn.Linear(encoder_output_size, zdim)
            self._proj_layer_source_vars = encoder_proj_layer.parameters()
            encoder_model = nn.Sequential(
                encoder_model,
                encoder_proj_layer,
                nn.ReLU,
            )

            target_proj_layer = nn.Linear(encoder_output_size, zdim)
            self._proj_layer_target_vars = target_proj_layer.parameters()
            target_encoder_model = nn.Sequential(
                target_encoder_model, target_proj_layer, nn.ReLU
            )

            # Update the projection layers on the target to match the source
            soft_update_from_to(
                self._proj_layer_source_vars, self._proj_layer_target_vars, tau=1.0
            )
        else:
            self._zdim = encoder_output_size

        self._encoder = encoder_model
        self._target_encoder = target_encoder_model
        self._W = torch.tensor(torch.rand(size=[zdim, zdim]), requires_grad=True)

        if self._use_temperature_variant:
            self._temperature = torch.tensor(
                temperature, requires_grad=False, dtype=torch.float32
            )

    # @property
    # def vars(self):
    #     return self._encoder.trainable_variables + [self._W]

    # def parameters(self):
    #     trainable_vars = [self._W]
    #     if self._use_projection_layer:
    #         trainable_vars += self._proj_layer_source_vars
    #         trainable_vars += self._proj_layer_target_vars
    #     return trainable_vars

    def _add_noise(self, image, random_number, std=0.025, prob=0.25):
        # Add noise to the image from a normal distribution.
        noise = torch.normal(mean=0, std=std, shape=image.shape)
        if (1.0 - prob) > random_number[0]:
            return image + noise
        else:
            return image

    # @tf.function
    def augment(self, image):
        if self._use_noise_aug:
            image = self._add_noise(
                image, rand_uniform(minval=0, maxval=1, shape=[1]), std=0.015
            )

        if self._use_fast_augment:
            image = self._cutout_per_batch_pos_and_mask_size(image)
            image = image.detach()
        else:
            if self._use_per_image_mask_size:
                image = self._cutout_per_image_mask_size(image)
            else:
                image = self._cutout_per_batch_mask_size(image)
            image = image.detach()
        return image

    # @tf.function
    def _loss(self, image1: torch.Tensor, image2: torch.Tensor):
        batch_size = image1.shape[0]

        # ENCODE
        z_a = self._encoder(image1)
        z_pos = self._target_encoder(image2)
        z_pos = z_pos.detach()

        # PROJECTION
        Wz = self._W @ z_pos.T  # (z,B)

        # LOGITS
        logits: torch.Tensor = z_a @ Wz  # (B,B)
        if self._use_temperature_variant:
            # Use normalised temperature scaled cross-entropy. This differs from the orig
            # CURL paper but it seems to give better results. This technique is also used
            # in SimCLR v2.
            logits = logits / self._temperature
        else:
            # remove max for numerical stability, as in original paper.
            logits = logits - torch.max(logits, dim=1)

        # LOSS
        # One neat trick!: Diags are positive examples, off-diag are negative examples!
        labels = F.one_hot(torch.range(batch_size), batch_size)
        loss = F.cross_entropy(labels, logits)
        return torch.mean(loss)

    # @tf.function
    def _cutout_per_image_mask_size(self, images):
        # This is slightly slower than per batch version but in principle it should also be slightly better.
        batch_size, im_x, im_y, im_ch = images.shape

        new_images = images.clone()
        for i in range(batch_size):
            mask_size_factor = torch.randint(low=2, high=4, size=[1])[0]
            mask_size = [int(im_x / mask_size_factor), int(im_y / mask_size_factor)]
            start_i = torch.randint(low=0, high=im_x - mask_size[0], size=[1])
            start_j = torch.randint(low=0, high=im_y - mask_size[1], size=[1])
            end_i = start_i + mask_size[0]
            end_j = start_j + mask_size[1]
            cutout = torch.zeros(
                [mask_size[0], mask_size[1], im_ch], dtype=torch.float32
            )
            new_images[i, start_i:end_i, start_j:end_j, :] = cutout
        return new_images

    # @tf.function
    def _cutout_per_batch_mask_size(self, images):
        batch_size, im_x, im_y, im_ch = images.shape

        mask_size_factor = torch.randint(low=2, high=4, size=[1])[0]
        mask_size = [int(im_x / mask_size_factor), int(im_y / mask_size_factor)]

        offset_i = torch.randint(low=0, high=im_x - mask_size[0], size=[batch_size])
        offset_j = torch.randint(low=0, high=im_y - mask_size[1], size=[batch_size])

        new_images = images.clone()
        for i in range(batch_size):
            start_i, start_j = offset_i[i], offset_j[i]
            end_i = start_i + mask_size[0]
            end_j = start_j + mask_size[1]
            cutout = torch.zeros(
                [mask_size[0], mask_size[1], im_ch], dtype=torch.float32
            )
            new_images[i, start_i:end_i, start_j:end_j, :] = cutout

        return new_images

    # @tf.function
    def _cutout_per_batch_pos_and_mask_size(self, images):
        batch_size, im_x, im_y, im_ch = images.shape

        mask_size_factor = torch.randint(low=2, high=4, size=[1])[0]
        mask_size = [int(im_x / mask_size_factor), int(im_y / mask_size_factor)]

        start_i = torch.randint(low=0, high=im_x - mask_size[0], size=[])
        start_j = torch.randint(low=0, high=im_y - mask_size[1], size=[])
        end_i = start_i + mask_size[0]
        end_j = start_j + mask_size[1]

        new_images = images.clone()
        cutout = torch.zeros([mask_size[0], mask_size[1], im_ch], dtype=torch.float32)
        new_images[:, start_i:end_i, start_j:end_j, :] = cutout
        return new_images

    def loss(self, observation):
        images = observation["obs"]
        image_aug1 = self.augment(images)
        if self._augment_anchor_and_pos:
            image_aug2 = self.augment(images)
        else:
            # This saves some computation and doesn't seem to have any adverse effects.
            image_aug2 = images

        if self._log_images:
            self.log_image("augmentations/base_allch", images[0:2, :, :, :])
            self.log_image("augmentations/image1_allch", image_aug1[0:2, :, :, :])
            self.log_image("augmentations/image2_allch", image_aug2[0:2, :, :, :])

        loss = self._loss(image_aug1, image_aug2)
        return loss

    def end_batch(self):
        if self._use_projection_layer:
            soft_update_from_to(
                self._proj_layer_source_vars, self._proj_layer_target_vars, tau=self._tau
            )
