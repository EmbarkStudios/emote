import argparse

import torch

from experiments.gym.train_lunar_lander import (
    _make_env,
    create_actor_critic_agents,
    create_complementary_callbacks,
    create_memory,
    create_train_callbacks,
)
from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from torch.optim import Adam

from emote import Trainer
from emote.extra.schedules import BPStepScheduler
from emote.models.callbacks import BatchSampler, ModelBasedCollector
from emote.models.ensemble import EnsembleOfGaussian
from emote.models.model import DynamicModel, ModelLoss
from emote.models.model_env import ModelEnv


def lunar_lander_term_func(
    states: torch.Tensor,
):
    """The termination function used to identify terminal states for the lunar lander
    gym environment. This function is used inside a gym-like dynamic model to terminate
    trajectories. The current implementation always outputs False which means all states
    labeled as non-terminal. This can be improved by adding some code to identify terminal
    states, or alternatively, training a neural network to detect terminal states.
        Arguments:
            states (torch.Tensor): the state (batch_size x dim_state)
        Returns:
            (torch.Tensor): the terminal labels (batch_size)
    """
    return torch.zeros(states.shape[0])


def create_dynamic_model_env(
    args,
    num_obs: int,
    num_actions: int,
):
    """Creates gym-like dynamic model
    Arguments:
        args: arguments passed to the code via argparse
        num_obs (int): the dimension of observation space
        num_actions (int): the dimension of action space
    Returns:
        (ModelEnv): Gym-like dynamic model
    """
    device = torch.device(args.device)
    model = EnsembleOfGaussian(
        in_size=num_obs + num_actions,
        out_size=num_obs + 1,
        device=device,
        ensemble_size=args.num_model_ensembles,
    )
    dynamic_model = DynamicModel(model=model)
    model_env = ModelEnv(
        num_envs=args.batch_size,
        model=dynamic_model,
        termination_fn=lunar_lander_term_func,
    )
    return model_env


def create_model_based_callbacks(
    args,
    model_buffer,
    model_data_loader,
    model_env,
    policy_proxy,
):
    """ "Creates the extra callbacks required for model-based RL (MBRL) training.
    Currently, there are three callbacks required for the MBRL training:
        (1) ModelLoss: It is used to train the dynamic model.
        (2) BatchSampler: In every BP step, it samples a batch of transitions from either the gym buffer or
        the model buffer depending on a probability distribution. The batch is only used for the RL training.
        (3) ModelBasedCollector: It is used to create synthetic transitions by unrolling the gym-like dynamic
        model. The transitions are stored in the model buffer.

        Arguments:
            args: arguments passed to the code via argparse
            model_buffer (DictTable): the replay_buffer used to store transitions
            model_data_loader (MemoryLoader): the dataloader used to sample batches of transitions
            model_env (ModelEnv): the Gym-like dynamic model
            policy_proxy (FeatureAgentProxy): the policy proxy
        Returns:
            (list[Callback]): A list of callbacks required for model-based RL training
    """
    mb_cbs = [
        ModelLoss(
            model=model_env.dynamic_model,
            name="dynamic_model",
            opt=Adam(model_env.dynamic_model.model.parameters(), lr=args.model_lr),
            data_group="default",
        ),
        BatchSampler(
            dataloader=model_data_loader,
            prob_scheduler=BPStepScheduler(*args.data_scheduler),
            data_group="default",
            rl_data_group="rl_buffer",
        ),
        ModelBasedCollector(
            model_env=model_env,
            agent=policy_proxy,
            memory=model_buffer,
            rollout_scheduler=BPStepScheduler(*args.rollout_scheduler),
            num_bp_to_retain_buffer=args.num_bp_to_retain_model_buffer,
        ),
    ]
    return mb_cbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ll_mbrl_")
    parser.add_argument("--log-dir", type=str, default="/mnt/mllogs/emote/lunar_lander")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--hidden-layer-size", type=int, default=256)
    parser.add_argument(
        "--actor-lr", type=float, default=8e-3, help="The policy learning rate"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=8e-3, help="Q-function learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bp-steps", type=int, default=100000)

    """The extra arguments for the model-based RL training"""
    parser.add_argument(
        "--num-model-ensembles",
        type=int,
        default=5,
        help="The number of dynamic models in the ensemble",
    )
    parser.add_argument(
        "--rollout-scheduler",
        nargs="+",
        type=int,
        default=[5000, 100000, 1, 20],
        help="The scheduler which outputs the rollout size (the number of time-steps"
        "to unroll the dynamic model) given the BP step as input "
        "[bp_begin, bp_end, rollout_initial_size, rollout_final_size]).",
    )
    parser.add_argument(
        "--data-scheduler",
        nargs="+",
        type=float,
        default=[5000, 100000, 0.0, 0.9],
        help="The scheduler which outputs the probability of choosing synthetic samples"
        "(generated by the model) against real gym samples given the BP step as input "
        "[bp_begin, bp_end, prob_initial_value, prob_final_value]).",
    )
    parser.add_argument(
        "--num-bp-to-retain-model-buffer",
        type=int,
        default=5000,
        help="The number of BP steps before the model-buffer is completely overwritten",
    )
    parser.add_argument(
        "--model-lr", type=float, default=1e-3, help="The model learning rate"
    )

    input_args = parser.parse_args()

    training_device = torch.device(input_args.device)

    gym_wrapper = DictGymWrapper(
        AsyncVectorEnv([_make_env() for _ in range(input_args.num_envs)])
    )
    number_of_actions = gym_wrapper.dict_space.actions.shape[0]
    number_of_obs = list(gym_wrapper.dict_space.state.spaces.values())[0].shape[0]

    """Creating the models, memory, dataloader and callbacks (the same as model-free training). """
    qnet1, qnet2, agent_proxy = create_actor_critic_agents(
        args=input_args, num_actions=number_of_actions, num_obs=number_of_obs
    )

    gym_memory, dataloader = create_memory(
        env=gym_wrapper,
        memory_size=4_000_000,
        len_rollout=input_args.rollout_length,
        batch_size=input_args.batch_size,
        data_group="default",
        device=training_device,
    )

    train_callbacks = create_train_callbacks(
        args=input_args,
        q1=qnet1,
        q2=qnet2,
        policy_proxy=agent_proxy,
        env=gym_wrapper,
        memory_proxy=gym_memory,
        data_group="rl_buffer",
    )

    """The extra functions used only for model-based RL training"""
    memory_init_size = input_args.batch_size * input_args.num_bp_to_retain_model_buffer
    model_memory, model_dataloader = create_memory(
        env=gym_wrapper,
        memory_size=memory_init_size,
        len_rollout=1,
        batch_size=input_args.batch_size,
        data_group="rl_buffer",
        device=training_device,
    )

    gym_like_env = create_dynamic_model_env(
        args=input_args,
        num_obs=number_of_obs,
        num_actions=number_of_actions,
    )

    mb_callbacks = create_model_based_callbacks(
        args=input_args,
        model_buffer=model_memory,
        model_data_loader=model_dataloader,
        model_env=gym_like_env,
        policy_proxy=agent_proxy,
    )

    """Creating the complementary callbacks and starting the training"""
    callbacks = create_complementary_callbacks(
        args=input_args, train_cbs=(train_callbacks + mb_callbacks)
    )

    trainer = Trainer(callbacks, dataloader)
    trainer.train()
