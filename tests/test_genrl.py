import numpy as np
import torch
import torch.types

from torch import nn

from emote.algorithms.genrl.proxies import MemoryProxyWithEncoder
from emote.algorithms.genrl.wrappers import DecoderWrapper, EncoderWrapper, PolicyWrapper
from emote.memory.builder import DictObsTable
from emote.nn.action_value_mlp import ActionValueMlp
from emote.nn.gaussian_policy import GaussianMlpPolicy
from emote.typing import DictObservation, DictResponse, EpisodeState
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


def get_conditioning_fn(len_cond: int = 0):
    assert len_cond >= 0

    def conditioning_fn(a):
        return a[:, :len_cond]

    return conditioning_fn


class FullyConnectedEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device,
        condition_size: int = 0,
        hidden_sizes: list[int] = None,
    ):
        super().__init__()
        self.device = device

        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size

        num_layers = len(hidden_sizes)
        layers = [
            nn.Sequential(
                nn.Linear(input_size + condition_size, hidden_sizes[0]),
                nn.ReLU(),
            )
        ]
        for i in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.ReLU(),
                )
            )
        self.hidden_layers = nn.Sequential(*layers).to(self.device)
        self.output_mean = nn.Linear(hidden_sizes[num_layers - 1], output_size).to(self.device)
        self.output_log_std = nn.Linear(hidden_sizes[num_layers - 1], output_size).to(self.device)

    def forward(
        self, data: torch.Tensor, condition: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if condition is not None:
            data = torch.cat((data, condition), dim=len(data.shape) - 1)

        x = self.hidden_layers(data)

        mean = self.output_mean(x)
        log_std = self.output_log_std(x)
        return mean, log_std


class FullyConnectedDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        device: torch.device,
        condition_size: int = 0,
        hidden_sizes: list[int] = None,
        freeze_bn: bool = True,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 256, 512]

        self.device = device

        self.input_size = latent_size
        self.output_size = output_size
        self.condition_size = condition_size
        self._freeze_bn = freeze_bn

        num_layers = len(hidden_sizes)
        layers = [
            nn.Sequential(
                nn.Linear(latent_size + condition_size, hidden_sizes[0]),
                nn.ReLU(),
            )
        ]
        for i in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.ReLU(),
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(hidden_sizes[num_layers - 1], output_size),
            )
        )
        self.layers = nn.Sequential(*layers).to(self.device)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if condition is not None:
            latent = torch.cat((latent, condition), dim=len(latent.shape) - 1)
        x = self.layers(latent)
        return x


LATENT_SIZE = 3
ACTION_SIZE = 2
OBSERVATION_SIZE = 24
CONDITION_SIZE = 24
BATCH_SIZE = 50

HIDDEN_SIZES = [256] * 2


def test_genrl():

    cfn = get_conditioning_fn(CONDITION_SIZE)
    device = torch.device("cpu")

    decoder = FullyConnectedDecoder(LATENT_SIZE, ACTION_SIZE, device, CONDITION_SIZE, HIDDEN_SIZES)
    decoder_wrapper = DecoderWrapper(decoder, cfn)
    encoder = FullyConnectedEncoder(ACTION_SIZE, LATENT_SIZE, device, CONDITION_SIZE, HIDDEN_SIZES)
    encoder_wrapper = EncoderWrapper(encoder, cfn)

    q = ActionValueMlp(OBSERVATION_SIZE, LATENT_SIZE, HIDDEN_SIZES).to(device)
    policy = GaussianMlpPolicy(OBSERVATION_SIZE, LATENT_SIZE, HIDDEN_SIZES).to(device)
    policy_wrapper = PolicyWrapper(decoder_wrapper, policy)

    obs = torch.rand(BATCH_SIZE, OBSERVATION_SIZE)

    action, log_prob = policy_wrapper.forward(obs)
    latent = encoder_wrapper.forward(action, obs)
    q_vals = q.forward(latent, obs)

    assert action.shape == (BATCH_SIZE, ACTION_SIZE)
    assert q_vals.shape == (BATCH_SIZE, 1)
    assert log_prob.shape == (BATCH_SIZE, 1)


def test_memory_proxy():

    cfn = get_conditioning_fn(CONDITION_SIZE)
    device = torch.device("cpu")

    encoder = FullyConnectedEncoder(ACTION_SIZE, LATENT_SIZE, device, CONDITION_SIZE, HIDDEN_SIZES)
    encoder_wrapper = EncoderWrapper(encoder, cfn)

    space = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=(ACTION_SIZE,)),
        state=DictSpace(spaces={"obs": BoxSpace(dtype=np.float32, shape=(OBSERVATION_SIZE,))}),
    )

    table = DictObsTable(spaces=space, maxlen=1000, device=device)

    proxy = MemoryProxyWithEncoder(
        table=table,
        encoder=encoder_wrapper,
        minimum_length_threshold=1,
        use_terminal=True,
    )

    agent_id = 0
    obs = np.random.rand(1, OBSERVATION_SIZE)
    action = np.random.rand(1, ACTION_SIZE)

    proxy.add(
        {
            agent_id: DictObservation(
                episode_state=EpisodeState.INITIAL,
                array_data={"obs": obs},
                rewards={"reward": None},
            )
        },
        {0: DictResponse({"actions": action}, {})},
    )

    assert (obs == proxy._store[agent_id].data["obs"][0]).all()
