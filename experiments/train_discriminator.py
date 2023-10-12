import os
import time
from typing import List
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter

from emote.memory.builder import DictObsNStepTable
from emote.memory.memory import MemoryLoader
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.nn.initialization import ortho_init_
from emote.callbacks import LossCallback
from emote.callbacks.logging import TensorboardLogger
from emote.callbacks.generic import BackPropStepsTerminator
from emote import Callback
from emote.callbacks.logging import LoggingMixin
from emote.trainer import Trainer


class Discriminator(nn.Module):
    def __init__(self, input_size: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([input_size] + hidden_dims, hidden_dims)
            ],
        )
        final_layers: list[nn.Module] = [nn.Linear(hidden_dims[-1], 1)]
        self.final_layer = nn.Sequential(*final_layers)

        self.encoder.apply(ortho_init_)
        self.final_layer.apply(ortho_init_)

    def forward(self, x: Tensor):
        return self.final_layer(self.encoder(x))


class ActionDiscriminatorReward(Callback, LoggingMixin):
    def __init__(
        self,
        discriminator: Discriminator,
        confusion_reward_weight: float,
        data_group: str,
        left_indices: List[int],
        right_indices: List[int],
        mirror_indices: List[int],
    ):
        self._order = 0
        super().__init__()
        self._discriminator = discriminator
        self._reward_weight = confusion_reward_weight
        self.data_group = data_group
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.mirror_indices = mirror_indices

    def begin_batch(self, actions: Tensor, rewards: Tensor):
        batch_size = actions.shape[0]

        actions_mirrored = torch.clone(actions)
        actions_mirrored[:, self.mirror_indices] = -1.0 * actions_mirrored[:, self.mirror_indices]

        actions_separated = torch.cat(
            (actions_mirrored[:, self.right_indices], actions_mirrored[:, self.left_indices]), dim=0
        )

        predictions = self._discriminator(actions_separated)
        predictions_right = predictions[:batch_size]  # labeled 1
        predictions_left = predictions[batch_size:]  # labeled -1

        predictions_right = torch.clamp(predictions_right, -1.0, 1.0)
        predictions_left = torch.clamp(predictions_left, -1.0, 1.0)

        confusion_reward = 1.0 - 0.25 * (predictions_left - 1.0) ** 2
        confusion_reward += 1.0 - 0.25 * (predictions_right + 1.0) ** 2

        scaled_confusion_reward = confusion_reward * self._reward_weight

        total_reward = rewards + scaled_confusion_reward

        self.log_scalar("amp/unscaled_confusion_reward", torch.mean(confusion_reward))
        self.log_scalar("amp/scaled_confusion_reward", torch.mean(scaled_confusion_reward))
        self.log_scalar("amp/task_reward", torch.mean(rewards))
        self.log_scalar("amp/total_reward", torch.mean(total_reward))

        return {self.data_group: {"rewards": total_reward}}


class ActionDiscriminatorLoss(LossCallback):
    def __init__(
            self,
            name: str,
            discriminator: Discriminator,
            optimizer: torch.optim.Optimizer,
            lr_schedule: torch.optim.lr_scheduler._LRScheduler,
            max_grad_norm: float,
            left_indices: List[int],
            right_indices: List[int],
            mirror_indices: List[int],
            device: torch.device,
            data_group: str = "default",
    ):
        super().__init__(
            lr_schedule=lr_schedule,
            name=name,
            network=discriminator,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self.device = device
        self._discriminator = discriminator
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.mirror_indices = mirror_indices

    def loss(self, actions):
        batch_size = actions.shape[0]
        actions_mirrored = torch.clone(actions)
        actions_mirrored[:, self.mirror_indices] = -1.0 * actions_mirrored[:, self.mirror_indices]

        actions_separated = torch.cat(
            (
                actions_mirrored[:, self.right_indices],
                actions_mirrored[:, self.left_indices]
            ),
            dim=0
        )
        targets = torch.cat(
            (
                torch.ones(batch_size, 1),
                -1.0 * torch.ones(batch_size, 1)
            )
        ).to(self.device)

        predictions = self._discriminator(actions_separated)

        right_predictions = predictions[:batch_size, :]
        left_predictions = predictions[batch_size:, :]

        loss = torch.mean(
            torch.square(
                predictions - targets
            )
        )

        self.log_scalar("amp/right_prediction", torch.mean(right_predictions))
        self.log_scalar("amp/left_prediction", torch.mean(left_predictions))

        return loss


memory_path = "/home/ali/data/capy/replay_buffer/ampaction/"

action_size = 26
input_shapes = {
    "features": {
        "shape": [294]
    }
}
data_group = "rl_loader"
device = torch.device('cuda:0')
batch_size = 500
rollout_length = 5

state_spaces = {
    k: BoxSpace(dtype=np.float32, shape=tuple(v["shape"]))
    for k, v in input_shapes.items()
}
spaces = MDPSpace(
    rewards=None,
    actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
    state=DictSpace(state_spaces),
)

table = DictObsNStepTable(
    spaces=spaces,
    use_terminal_column=True,
    maxlen=1_000_000,
    device=device,
)
restore_path = os.path.join(
    memory_path, f"{data_group}_export"
)
table.restore(restore_path)
print(f"the size of the table is: {table.size()}")

data_loader = MemoryLoader(
    table,
    batch_size // rollout_length,
    rollout_length,
    "batch_size",
    data_group=data_group,
)

mirrored_indices = []  # [7, 8, 20, 21]
right_action_indices = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17]
left_action_indices = [5, 6, 7, 8, 9, 18, 19, 20, 21, 22]
hidden_layers = [512] * 4

discriminator_lr_init = 0.00005
discriminator_lr_end = 0.00001
discriminator_lr_steps = 300000

bp_steps = 10000

discriminator = Discriminator(len(right_action_indices), hidden_layers)
discriminator = discriminator.to(device)
discriminator_opt = torch.optim.Adam(
    discriminator.parameters(), lr=discriminator_lr_init
)
discriminator_schedule = LinearLR(
    discriminator_opt,
    1.0,
    discriminator_lr_end / discriminator_lr_init,
    discriminator_lr_steps,
)

training_cbs = [
    ActionDiscriminatorLoss(
        name='discriminator',
        discriminator=discriminator,
        optimizer=discriminator_opt,
        lr_schedule=discriminator_schedule,
        max_grad_norm=1.0,
        left_indices=left_action_indices,
        right_indices=right_action_indices,
        mirror_indices=mirrored_indices,
        device=device,
        data_group=data_group,
    ),
    ActionDiscriminatorReward(
        discriminator=discriminator,
        confusion_reward_weight=0.5,
        data_group=data_group,
        left_indices=left_action_indices,
        right_indices=right_action_indices,
        mirror_indices=mirrored_indices
    )
]

logger = TensorboardLogger(
    training_cbs,
    SummaryWriter(log_dir="logs" + "/discriminator/training" + "_{}".format(time.time())),
    100,
)
bp_step_terminator = BackPropStepsTerminator(bp_steps=bp_steps)

callbacks = training_cbs + [logger, bp_step_terminator]

trainer = Trainer(callbacks=callbacks, dataloader=data_loader)
trainer.train()
