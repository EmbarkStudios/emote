from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FrozenBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        bn_training = False
        if len(input.shape) == 1:
            input = input.unsqueeze(dim=0)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        if input.dim() != 1 and input.dim() != 2:
            raise ValueError("expected 1D or 2D input (got {}D input)".format(input.dim()))


class FullyConnectedMapping(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device,
        gaussian_head: bool,
        condition_size: int = 0,
        hidden_sizes: list[int] = None,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.device = device
        self.gaussian_head = gaussian_head
        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size
        self._freeze_bn = freeze_bn

        num_layers = len(hidden_sizes)
        batch_norm = (
            FrozenBatchNorm1d(hidden_sizes[0])
            if self._freeze_bn
            else nn.BatchNorm1d(hidden_sizes[0])
        )
        layers = [
            nn.Sequential(
                nn.Linear(input_size + condition_size, hidden_sizes[0]),
                batch_norm,
                nn.ReLU(),
            )
        ]
        for i in range(num_layers - 1):
            batch_norm = (
                FrozenBatchNorm1d(hidden_sizes[i + 1])
                if self._freeze_bn
                else nn.BatchNorm1d(hidden_sizes[i + 1])
            )
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    batch_norm,
                    nn.ReLU(),
                )
            )
        if self.gaussian_head:
            self.output_mean = nn.Linear(hidden_sizes[num_layers - 1], output_size).to(self.device)
            self.output_log_std = nn.Linear(hidden_sizes[num_layers - 1], output_size).to(
                self.device
            )
        else:
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[num_layers - 1], output_size),
                )
            )
        self.layers = nn.Sequential(*layers).to(self.device)

    def forward(
        self, data: torch.Tensor, condition: torch.Tensor = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if condition is not None:
            data = torch.cat((data, condition), dim=len(data.shape) - 1)

        x = self.layers(data)
        if self.gaussian_head:
            return x

        mean = self.output_mean(x)
        log_std = self.output_log_std(x)
        return mean, log_std


from emote.algorithms.genrl.wrappers import DecoderWrapper, EncoderWrapper


device = torch.device('cpu')
decoder = FullyConnectedMapping(
    input_size=5,
    output_size=28,
    device=device,
    condition_size=168,
    gaussian_head=False,
    hidden_sizes=[1024] * 4,
)
encoder = FullyConnectedMapping(
    input_size=28,
    output_size=5,
    device=device,
    condition_size=168,
    gaussian_head=True,
    hidden_sizes=[1024] * 4,
)
decoder_wrapper = DecoderWrapper(
    decoder=decoder,
    condition_fn=None,
)
encoder_wrapper = EncoderWrapper(
    encoder=encoder,
    condition_fn=None,
)
state = decoder_wrapper.state_dict()
state.update(encoder_wrapper.state_dict())

network_dict = {"network_state_dict": state}
state_dict = {"callback_state_dicts": {"vae": network_dict}}

torch.save(state_dict, "vae_temp.tar")

state_dict = torch.load("vae_temp.tar", map_location=device)
state = state_dict["callback_state_dicts"]["vae"]
decoder_wrapper.load_state_dict(state["network_state_dict"])
encoder_wrapper.load_state_dict(state["network_state_dict"])

