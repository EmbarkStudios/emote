import numpy as np
import torch
from torch import nn
import argparse
from emote.nn.layers import PointCloudPolicy, PointNetEncoder, PointCloudQNet
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.memory import MemoryLoader
from emote.utils.gamma_matrix import discount, make_gamma_matrix, split_rollouts
from torch.optim import Adam

from torch.profiler import profile, record_function, ProfilerActivity


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch", type=int, default=4000)
    parser.add_argument("--action-dim", type=int, default=22)
    parser.add_argument("--observation-dim", type=int, default=225)
    parser.add_argument("--point-count", type=int, default=300)
    parser.add_argument("--point-dim", type=int, default=3)
    parser.add_argument("--pointnet-output-dim", type=int, default=128)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--memory-size", type=int, default=500000)
    parser.add_argument("--policy-hidden-sizes", nargs="+", type=int, default=[1024, 1024])
    parser.add_argument("--pointnet-input-hidden-sizes", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--pointnet-feature-hidden-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--use-terminal", action='store_true')
    parser.add_argument("--use-batch-norm", action='store_true')
    parser.add_argument("--use-tnet", action='store_true')

    arg = parser.parse_args()

    memory_path = arg.path_to_buffer.replace('.zip', '')
    device = torch.device(arg.device)

    obs_key = 'features'
    pc_key = 'point_cloud'

    action_size = arg.action_dim
    input_shapes = {
        obs_key: {
            "shape": [arg.observation_dim]
        },
        pc_key: {
            "shape": [arg.point_count * arg.point_dim]
        }
    }

    state_spaces = {
        k: BoxSpace(dtype=np.float32, shape=tuple(v["shape"]))
        for k, v in input_shapes.items()
    }

    input_spaces = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
        state=DictSpace(state_spaces),
    )

    table = DictObsNStepTable(
        spaces=input_spaces,
        use_terminal_column=arg.use_terminal,
        maxlen=arg.memory_size,
        device=device,
    )

    table.restore(memory_path)
    print(f"the size of the table is: {table.size()}")
    data_loader = MemoryLoader(
        table,
        arg.batch // arg.rollout_length,
        arg.rollout_length,
        "batch_size",
        data_group="group",
    )

    pc_encoder = PointNetEncoder(
        num_points=arg.point_count,
        input_dim=arg.point_dim,
        output_dim=arg.pointnet_output_dim,
        input_stack_hidden_dims=arg.pointnet_input_hidden_sizes,
        feature_stack_hidden_dims=arg.pointnet_feature_hidden_sizes,
        use_batch_norm=arg.use_batch_norm,
        use_tnet=arg.use_tnet,
        device=device,
    )
    policy = PointCloudPolicy(
        shared_enc=pc_encoder,
        action_dim=arg.action_dim,
        hidden_dims=arg.policy_hidden_sizes,
        observation_dim=arg.observation_dim + arg.pointnet_output_dim,
    )
    qnet = PointCloudQNet(
        shared_enc=pc_encoder,
        action_dim=arg.action_dim,
        hidden_dims=arg.policy_hidden_sizes,
        obs_dim=arg.observation_dim + arg.pointnet_output_dim
    )
    alpha = 0.5
    reward_scale = 1.0
    gamma = 0.97
    ln_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
    policy = policy.to(device)
    qnet = qnet.to(device)

    q_opt = Adam(qnet.parameters(), lr=0.001)
    p_opt = Adam(policy.parameters(), lr=0.001)

    mse_loss = nn.MSELoss()

    gamma_matrix = make_gamma_matrix(gamma, arg.rollout_length).to(device)

    itr = iter(data_loader)
    data = next(itr)

    observation = data['group']['observation']
    next_observation = data['group']['next_observation']
    actions = data['group']['actions']
    rewards = data['group']['rewards']
    masks = data['group']['masks']
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("training"):

            # compute next q
            next_p_sample, next_log_p = policy(**next_observation)
            min_next_qt = qnet(next_p_sample, **next_observation)
            bsz = rewards.shape[0]
            alpha = torch.exp(ln_alpha)
            next_value = min_next_qt - alpha * next_log_p
            scaled_reward = reward_scale * rewards

            last_step_masks = split_rollouts(masks, arg.rollout_length)[:, -1]
            scaled_reward = split_rollouts(scaled_reward, arg.rollout_length).squeeze(2)

            if arg.use_terminal:
                next_value = torch.multiply(next_value, last_step_masks)

            q_target = discount(scaled_reward, next_value, gamma_matrix).detach()

            # compute the q-loss
            q_opt.zero_grad()

            q_value = qnet(actions, **observation)
            q_loss = mse_loss(q_value, q_target)

            # optimize the q-loss
            q_loss.backward()
            q_opt.step()

            # compute policy loss
            p_opt.zero_grad()

            p_sample, log_p = policy(**observation)
            q_pi_min = qnet(p_sample, **observation)
            alpha = torch.exp(ln_alpha).detach()
            policy_loss = alpha * log_p - q_pi_min
            policy_loss = torch.mean(policy_loss)

            # optimize the policy loss
            policy_loss.backward()
            p_opt.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
