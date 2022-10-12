"""
Utilities for loading files into memories.
"""

import pickle

import numpy as np

from .table import ArrayTable


def fill_table_from_legacy_file(
    table: ArrayTable,
    path: str,
    *,
    read_obs: bool = False,
    read_actions: bool = False,
    read_rewards: bool = False,
):
    """Load a legacy memory dump into a new-style table memory.

    :param table: The table to fill. Must contain 'obs', 'rewards', and 'actions' columns
    :param path: The path to load from. Must be a pickle file. Extension is optional

    :throws: OSError if file does not exist. KeyError if table or file do not
    match the legacy format.
    """

    if not path.endswith(".pickle"):
        path += ".pickle"

    with open(path, "rb") as file_:
        state = pickle.load(file_)  # nosec B301

    for k in ["dones", "actions", "rewards", "next_obs", "obs"]:
        array = np.array(state[k])
        state[k] = array.reshape(-1, *array.shape[3:])

    done_indices = [i for i, d in enumerate(state["dones"]) if d]
    previous_idx = 0
    agent_idx = -1
    for done_idx in done_indices:
        if (done_idx - previous_idx) < 10:
            previous_idx = done_idx + 1
            agent_idx -= 1
            continue

        rewards = state["rewards"][previous_idx : done_idx + 1]
        actions = state["actions"][previous_idx * 2 : (done_idx + 1) * 2]
        dones = state["dones"][previous_idx : done_idx + 1]

        assert not dones[0] or len(dones) == 1
        assert dones[len(dones) - 1]
        obs = [o for o in state["obs"][previous_idx : done_idx + 1]]

        obs.append(state["next_obs"][done_idx + 1])

        outs = {}
        if read_obs:
            outs["obs"] = obs

        if read_actions:
            outs["actions"] = actions.reshape(-1, 2)

        if read_rewards:
            outs["rewards"] = rewards

        table.add_sequence(agent_idx, outs)

        previous_idx = done_idx + 1
        agent_idx -= 1
