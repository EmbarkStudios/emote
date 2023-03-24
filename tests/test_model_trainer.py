import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from emote.models.model import DynamicModel, DeterministicModel
from emote.models.ensemble import EnsembleOfGaussian
from emote.models.callbacks import ModelLoss, LossProgressCheck
from gymnasium.vector import AsyncVectorEnv
from emote import Trainer
from emote.callbacks import TerminalLogger
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable, DictObsNStepTable
from emote.sac import FeatureAgentProxy
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector


def term_func(
        states: torch.Tensor,
):
    return torch.zeros(states.shape[0])


class RandomPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        rand_actions = 2 * (torch.rand(batch_size, self.action_dim) - 0.5)
        return rand_actions, 0


class HitTheMiddleModel(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device(device)
        self.net = nn.Linear(in_size, out_size).to(device)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        #print('x: ', x)

        batch_size = x.shape[0]
        prediction = torch.zeros(batch_size, 3)
        y = self.net(x)
        for i in range(batch_size):
            pos, vel, action = x[i, 0].clone(), x[i, 1].clone(), x[i, 2].clone()
            vel += action
            pos += vel

            if pos > 10.0:
                pos = 10.0
                vel *= -1.0
            elif pos < -10.0:
                pos = -10.0
                vel *= -1.0
            reward = -(pos**2)
            prediction[i, :] = torch.Tensor([pos, vel, reward])
        prediction = prediction.to(self.device)
        prediction += 0.0000000001*y
        #print('prediction: ', prediction)
        return prediction.to(self.device)

    def loss(
            self,
            model_in: torch.Tensor,
            target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        prediction = self.forward(model_in)
        loss = F.mse_loss(prediction, target)
        #print('model_in', model_in)
        #print('target', target)
        #print('prediction', prediction)
        return loss, {'loss_info': None}

    def sample(
        self,
        model_input: torch.Tensor,
        rng: torch.Generator = None,
    ) -> torch.Tensor:
        """Samples next observation, reward and terminal from the model

        Args:
            model_input (tensor): the observation and action.
            rng (torch.Generator): a random number generator.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary.
        """
        return self.forward(model_input)


def test_model_learning():
    device = torch.device("cuda")
    batch_size = 5
    rollout_length = 1
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    num_obs = 2
    num_actions = 1
    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=10000,
        device=device,
    )
    #table = DictObsTable(spaces=env.dict_space, maxlen=10000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table=table, rollout_count=batch_size // rollout_length,
                              rollout_length=rollout_length, size_key="batch_size")

    # model = EnsembleOfGaussian(
    #    in_size=num_obs + num_actions,
    #    out_size=num_obs + 1,
    #    device=device,
    #    ensemble_size=5,
    # )
    model = DeterministicModel(
        in_size=num_obs + num_actions,
        out_size=num_obs + 1,
        num_hidden_layers=2,
        device=device,
    )
    model = HitTheMiddleModel(
        in_size=num_obs + num_actions,
        out_size=num_obs + 1,
        device=device,
    )
    dynamic_model = DynamicModel(model=model, no_delta_list=[0,1])
    policy = RandomPolicy(action_dim=1)
    agent_proxy = FeatureAgentProxy(policy, device)

    logged_cbs = [
        ModelLoss(model=dynamic_model, opt=Adam(dynamic_model.model.parameters())),
        LossProgressCheck(model=dynamic_model, num_bp=500),
    ]
    callbacks = logged_cbs + [
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
        TerminalLogger(logged_cbs, 400),
    ]
    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()


if __name__ == "__main__":
    test_model_learning()