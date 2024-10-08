import argparse
import math
import random
import time

import gymnasium as gym
import numpy as np
import torch

from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.algorithms.dqn import QLoss, QTarget
from emote.callbacks.checkpointing import Checkpointer
from emote.callbacks.generic import BackPropStepsTerminator
from emote.callbacks.logging import TensorboardLogger
from emote.memory import MemoryLoader, MemoryTableProxy
from emote.memory.builder import DictObsNStepMemoryTable
from emote.mixins.logging import LoggingMixin
from emote.proxies import GenericAgentProxy
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


def _make_env():
    """Create the environment for the experiment, the environment is created in
    a thunk to avoid creating multiple environments in the same process.

    This is important for the vectorized
    environments.
    Returns:
        (Callable[[], gym.Env]): The thunk that creates the environment
    """

    def _thunk():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.FrameStack(env, 3)
        env = gym.wrappers.FlattenObservation(env)
        return env

    return _thunk


class QNet(nn.Module):
    """Q-Network class for Q-Learning. It takes observations and returns
    Q-values for actions.

    Attributes:
        network (nn.Sequential): Neural network for computing Q-values.

    Args:
        num_obs (int): Dimensionality of observations.
        num_actions (int): Number of possible actions.
        hidden_dims (list of int): Dimensions of hidden layers.
    """

    def __init__(self, num_obs, num_actions, hidden_dims):
        super(QNet, self).__init__()

        layers = []
        input_dim = num_obs

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """Forward pass for the Q-Network.

        Args:
            obs (Tensor): Observations.

        Returns:
            Tensor: Q-values for each action.
        """
        return self.network(obs)


class DQNPolicy(nn.Module):
    """DQN Policy class to handle action selection with epsilon-greedy
    strategy.

    Attributes:
        q_net (QNet): Q-Network to evaluate Q-values.
        initial_epsilon (float): Initial value of epsilon in epsilon-greedy.
        target_epsilon (float): Target value of epsilon.
        step_count (int): Counter for steps taken.
        epsilon_decay_duration (int): Steps over which epsilon is decayed.
        log_epsilon (bool): Flag to log epsilon values.

    Args:
        q_net (QNet): Q-Network.
        epsilon_range (list of float): Initial and target epsilon for epsilon-greedy.
        epsilon_decay_duration (int): Number of steps over which epsilon will decay.
        log_epsilon (bool): Whether to log epsilon values or not.
    """

    def __init__(
        self, q_net, epsilon_range=[0.9, 0.05], epsilon_decay_duration=10_000, log_epsilon=True
    ):
        super(DQNPolicy, self).__init__()
        self.q_net = q_net

        self.initial_epsilon = epsilon_range[0]
        self.target_epsilon = epsilon_range[1]
        self.step_count = 0
        self.epsilon_decay_duration = epsilon_decay_duration
        self.log_epsilon = log_epsilon

    # Returns the index of the chosen action
    def forward(self, state):
        """Forward pass for action selection.

        Args:
            state (Tensor): The state observations.

        Returns:
            Tensor: Indices of chosen actions for each environment.
        """
        with torch.no_grad():
            epsilon = self.target_epsilon + (self.initial_epsilon - self.target_epsilon) * math.exp(
                -1.0 * self.step_count / self.epsilon_decay_duration
            )

            self.step_count += 1
            if (
                self.step_count % 50_000 == 0
                and self.log_epsilon
                and epsilon > self.target_epsilon + 0.01
            ):
                print("Epsilon: ", epsilon)

            q_values = self.q_net(state)  # Shape should be (num_envs, action_dim)
            num_envs, action_dim = q_values.shape
            actions = []

            for i in range(num_envs):
                if np.random.rand() < epsilon:
                    action_idx = random.randint(0, action_dim - 1)
                else:
                    action_idx = q_values[i].argmax().item()
                actions.append(action_idx)
            return torch.tensor(actions)


def create_memory(
    space: MDPSpace,
    memory_size: int,
    len_rollout: int,
    batch_size: int,
    data_group: str,
    device: torch.device,
):
    """Creates memory and data_loader for the RL training.

    Arguments:
        space (MDPSpace): the MDP space
        memory_size (int): the maximum length of memory
        len_rollout (int): the rollout size for the NStepTable
        batch_size (int): batch size
        data_group (str): the data group for uploading the data
        device (torch.device): the device to upload the data
        preload_buffer (bool): preload the buffer with some existing data
        buffer_filename (str): the path to the replay buffer if preload_buffer is set to True
    Returns:
        (tuple[MemoryTableProxy, MemoryLoader]): A proxy for the memory and a dataloader
    """
    # Create the memory
    memory_table = DictObsNStepMemoryTable(
        spaces=space,
        use_terminal_column=False,
        maxlen=memory_size,
        device=device,
    )
    # The memory proxy is used to upload the data to the memory
    memory_proxy = MemoryTableProxy(memory_table=memory_table, use_terminal=False)
    # The data loader is used to sample the data from the memory
    data_loader = MemoryLoader(
        memory_table=memory_table,
        rollout_count=batch_size // len_rollout,
        rollout_length=len_rollout,
        size_key="batch_size",
        data_group=data_group,
    )
    return memory_proxy, data_loader


def create_complementary_callbacks(
    args,
    logged_cbs: list[LoggingMixin],
    cbs_name_to_checkpoint: list[str] = None,
):
    """The function creates the supplementary callbacks for the training and
    adds them to the callback lists and returns the list.

    Arguments:
        args: input args
        logged_cbs (list[Callback]): the list of callbacks
        cbs_name_to_checkpoint (list[str]): the name of callbacks to checkpoint
    Returns:
        (list[Callback]): the full list of callbacks for the training
    """
    # The logger callback is used for logging the training progress
    logger = TensorboardLogger(
        logged_cbs,
        SummaryWriter(log_dir=args.log_dir + "/" + args.name + "_{}".format(time.time())),
        100,
    )

    # Terminates the training after a certain number of backprop steps
    bp_step_terminator = BackPropStepsTerminator(bp_steps=args.bp_steps)
    # Callbacks to be used during training
    callbacks = logged_cbs + [logger, bp_step_terminator]

    if cbs_name_to_checkpoint:
        # The checkpointer exports the model weights to the checkpoint directory
        checkpointer = Checkpointer(
            restorees=[
                cb for cb in logged_cbs if hasattr(cb, "name") and cb.name in cbs_name_to_checkpoint
            ],
            run_root=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
        )
        callbacks += [checkpointer]

    return callbacks


def main(args):
    # Create the environment
    env = DictGymWrapper(AsyncVectorEnv([_make_env() for _ in range(args.num_envs)]))
    device = torch.device(args.device)

    # Define the space in order to create the memory
    input_shapes = {k: v.shape for k, v in env.dict_space.state.spaces.items()}
    output_shapes = {"actions": env.dict_space.actions.shape}
    action_shape = output_shapes["actions"]
    spaces = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=action_shape),
        state=DictSpace(
            spaces={k: BoxSpace(dtype=np.float32, shape=tuple(v)) for k, v in input_shapes.items()}
        ),
    )
    num_actions = spaces.actions.shape[0]
    num_obs = list(spaces.state.spaces.values())[0].shape[0]

    memory_proxy, dataloader = create_memory(
        space=spaces,
        memory_size=args.memory_size,
        len_rollout=args.rollout_length,
        batch_size=args.batch_size,
        data_group="default",
        device=device,
    )
    """Create a memory exporter if needed."""
    if args.export_memory:
        from emote.memory.memory import MemoryExporterProxyWrapper

        memory_proxy = MemoryExporterProxyWrapper(
            memory=memory_proxy,
            target_memory_name=dataloader.data_group,
            inf_steps_per_memory_export=10_000,
            experiment_root_path=args.log_dir,
            min_time_per_export=0,
        )

    num_actions = env.action_space.nvec[0]

    # Create our two networks and the policy
    online_q_net = QNet(num_obs, num_actions, args.hidden_dims)
    target_q_net = QNet(num_obs, num_actions, args.hidden_dims)
    policy = DQNPolicy(online_q_net)

    # Move them to the device
    online_q_net = online_q_net.to(device)
    target_q_net = target_q_net.to(device)
    policy = policy.to(device)

    # The agent proxy is responsible for inference
    agent_proxy = GenericAgentProxy(
        policy,
        device=device,
        input_keys=tuple(input_shapes.keys()),
        output_keys=tuple(output_shapes.keys()),
        uses_logprobs=False,
        spaces=spaces,
    )

    # Create an optimizer for the online network
    optimizers = [
        QLoss(
            name="q1",
            q=online_q_net,
            opt=Adam(online_q_net.parameters(), lr=args.lr),
            max_grad_norm=1,
        ),
    ]

    train_callbacks = optimizers + [
        # The QTarget callback is responsible for updating the target network
        QTarget(
            q_net=online_q_net,
            target_q_net=target_q_net,
            roll_length=args.rollout_length,
        ),
        # The collector is responsible for the interaction with the environment
        ThreadedGymCollector(
            env,
            agent_proxy,
            memory_proxy,
            warmup_steps=args.batch_size * 2000,
            render=False,
        ),
    ]

    all_callbacks = create_complementary_callbacks(
        args,
        train_callbacks,
    )

    # The trainer acts as the main callback, responsible for calling all other callbacks
    trainer = Trainer(all_callbacks, dataloader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="cartpole", help="The name of the experiment")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./mllogs/emote/cartpole",
        help="Directory where logs will be stored.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of environments to run in parallel"
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=1,
        help="The length of each rollout. Refers to the number of steps or time-steps taken during a simulated trajectory or rollout when estimating the expected return of a policy.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Size of each training batch")
    parser.add_argument(
        "--hidden-dims", type=list, default=[128, 128], help="The hidden dimensions of the network"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on, e.g. cpu or cuda:0"
    )
    parser.add_argument(
        "--bp-steps",
        type=int,
        default=50_000,
        help="Number of backpropagation steps until the training run is finished",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=50_000,
        help="The size of the replay buffer. More complex environments require larger replay buffers, as they need more data to learn. Given that cartpole is a simple environment, a replay buffer of size 50_000 is sufficient.",
    )
    parser.add_argument(
        "--export-memory", action="store_true", default=False, help="Whether to export the memory"
    )
    args = parser.parse_args()
    main(args)


# pdm run python experiments/gym/train_dqn_cartpole.py
