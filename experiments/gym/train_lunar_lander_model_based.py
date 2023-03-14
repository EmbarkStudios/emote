import torch
import argparse
from torch.optim import Adam
from gymnasium.vector import AsyncVectorEnv

from emote import Trainer
from emote.models.ensemble import EnsembleOfGaussian
from emote.models.model import DynamicModel, ModelLoss
from emote.models.model_env import ModelEnv, BatchSampler, ModelBasedCollector
from emote.typing import BPStepScheduler
from tests.gym import DictGymWrapper
from experiments.gym.train_lunar_lander import create_memory, create_actor_critic_agents
from experiments.gym.train_lunar_lander import create_train_callbacks, create_full_callbacks, _make_env


def termination_func(states, _):
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
        ensemble_size=args.num_model_ensembles
    )
    dynamic_model = DynamicModel(model=model)
    model_env = ModelEnv(
        num_envs=args.batch_size,
        model=dynamic_model,
        termination_fn=termination_func,
    )
    return model_env


def create_model_based_callbacks(
        args,
        model_buffer,
        model_data_loader,
        model_env,
        policy_proxy,
):
    """"Creates the extra callbacks required for model-based RL training
        Arguments:
            args: arguments passed to the code via argparse
            model_buffer (DictTable): the replay_buffer used to store transitions
            model_data_loader (MemoryLoader): the dataloader used to sample batches of transitions
            model_env (ModelEnv): the Gym-like dynamic model
            policy_proxy (FeatureAgentProxy): the policy proxy
        Returns:
            (list[callbacks]): A list of callbacks required for model-based RL training
    """
    mb_cbs = [
        ModelLoss(
            model=model_env.dynamic_model,
            name='dynamic_model',
            opt=Adam(model_env.dynamic_model.model.parameters(), lr=args.model_lr),
            data_group='default',
        ),
        BatchSampler(
            dataloader=model_data_loader,
            model_data_prob_schedule=BPStepScheduler(*args.data_scheduler),
            data_group="rl_buffer",
        ),
        ModelBasedCollector(
            model_env=model_env,
            agent=policy_proxy,
            memory=model_buffer,
            rollout_scheduler=BPStepScheduler(*args.rollout_scheduler),
            num_bp_to_retain_buffer=args.num_bp_to_retain_sac_buffer,
        )
    ]
    return mb_cbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ll")
    parser.add_argument("--log-dir", type=str, default="/mnt/mllogs/emote/lunar_lander")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--actor-lr", type=float, default=8e-3, help='The policy learning rate')
    parser.add_argument("--critic-lr", type=float, default=8e-3, help='Q-function learning rate')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bp-steps", type=int, default=10000)

    parser.add_argument("--num-model-ensembles", type=int, default=5,
                        help='The number of dynamic models in the ensemble')
    parser.add_argument("--rollout-scheduler", type=list, default=[1000, 100000, 1, 20])
    parser.add_argument("--data-scheduler", type=list, default=[1000, 10000, 0.0, 0.9])
    parser.add_argument("--num-bp-to-retain-sac-buffer", type=int, default=5000)
    parser.add_argument("--model-lr", type=float, default=1e-3, help='The model learning rate')
    input_args = parser.parse_args()

    training_device = torch.device(input_args.device)

    gym_wrapper = DictGymWrapper(AsyncVectorEnv([_make_env() for _ in range(input_args.num_envs)]))
    number_of_actions = gym_wrapper.dict_space.actions.shape[0]
    number_of_obs = list(gym_wrapper.dict_space.state.spaces.values())[0].shape[0]

    qnet1, qnet2, agent_proxy = create_actor_critic_agents(input_args, number_of_actions, number_of_obs)

    gym_memory, dataloader = create_memory(env=gym_wrapper,
                                           memory_size=4_000_000,
                                           len_rollout=input_args.len_rollout,
                                           batch_size=input_args.batch_size,
                                           data_group='default',
                                           device=training_device)

    train_callbacks = create_train_callbacks(input_args, qnet1, qnet2, agent_proxy, gym_wrapper, gym_memory)

    # Now add model stuff
    memory_init_size = (
            input_args.batch_size *
            input_args.num_bp_to_retain_sac_buffer
    )
    model_memory, model_dataloader = create_memory(env=gym_wrapper,
                                                   memory_size=memory_init_size,
                                                   len_rollout=1,
                                                   batch_size=input_args.batch_size,
                                                   data_group='rl_buffer',
                                                   device=training_device)

    gym_like_env = create_dynamic_model_env(
        args=input_args,
        num_obs=number_of_obs,
        num_actions=number_of_actions,
    )

    mb_callbacks = create_model_based_callbacks(args=input_args,
                                                model_buffer=model_memory,
                                                model_data_loader=model_dataloader,
                                                model_env=gym_like_env,
                                                policy_proxy=agent_proxy)

    callbacks = create_full_callbacks(input_args, train_callbacks + mb_callbacks)

    trainer = Trainer(callbacks, dataloader)
    trainer.train()
