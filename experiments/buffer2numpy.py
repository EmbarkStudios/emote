import numpy as np
import torch
import argparse
import os
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--path-to-save", type=str, default="/home/ali/data/biped/figure/")
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=300)

    arg = parser.parse_args()

    memory_path = arg.path_to_buffer
    path_to_save = arg.path_to_save

    seq_length = arg.sequence_length
    minimum_samples = arg.min_samples

    observation_key = 'features'
    max_tries = 10

    action_size = 51
    input_shapes = {
        "features": {
            "shape": [252]
        }
    }
    device = torch.device('cpu')
    memory_max_length = 100_000

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
        use_terminal_column=True,
        maxlen=memory_max_length,
        device=device,
    )

    table.restore(memory_path)
    print(f"the size of the table is: {table.size()}")

    num_samples_collected = 0
    num_tries = 0

    while num_samples_collected < minimum_samples and num_tries < max_tries:
        try:
            samples = table.sample(count=1, sequence_length=seq_length)
            if num_samples_collected == 0:
                observations = samples['observation'][observation_key]
                actions = samples['actions']
            else:
                observations = torch.cat((observations, samples['observation'][observation_key]), 0)
                actions = torch.cat((actions, samples['actions']), dim=0)
            num_samples_collected += observations.shape[0]

        except:
            num_tries += 1
            print('problem with sampling')

    np_observations = observations.numpy()
    np_actions = actions.numpy()

    print(np_observations.shape)
    print(np_actions.shape)
    np.save(os.path.join(path_to_save, "actions.npy"), np_actions)
    np.save(os.path.join(path_to_save, "observations.npy"), np_observations)
