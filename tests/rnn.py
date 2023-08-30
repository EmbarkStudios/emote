import numpy as np
import pytest
import torch

from emote.nn.rnn import BurnInSamplerAdaptor


def test_adaptor_start_at_zero(mock_table):
    """
    When the sampling point is at zero the burn in lengths should be 0 and the burn in data
    should be zeros.
    """
    adaptor = BurnInSamplerAdaptor(["obs", "action"], 2)
    adaptor._table = mock_table

    result = adaptor(mock_results(0), 2, 5)

    assert np.all(result["burn_in_lengths"].numpy() == [0, 0])
    assert np.all(result["burn_in_masks"].numpy() == [0, 0])
    assert np.all(result["burn_in_obs"].shape == (4, 10))
    assert np.all(result["burn_in_obs"].numpy() == np.zeros((4, 10)))
    assert np.all(result["burn_in_action"].shape == (4, 5))
    assert np.all(result["burn_in_action"].numpy() == np.zeros((4, 5)))
    assert np.all(
        result["obs"].numpy()
        == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        ]
    )
    assert np.all(
        result["action"].numpy()
        == [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39],
        ]
    )


def test_adaptor_start_at_one(mock_table):
    """When the sampling point is at random the burn in lengths should be the actual length of the
    burn in and the burn in data should be the prefix of the regular data.

    """
    adaptor = BurnInSamplerAdaptor(["obs", "action"], 2)
    adaptor._table = mock_table

    result = adaptor(mock_results(1), 2, 5)

    assert np.all(result["burn_in_lengths"].numpy() == [1, 1])
    assert np.all(result["burn_in_masks"].numpy() == [1, 1])
    assert np.all(
        result["burn_in_obs"].numpy()
        == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    assert np.all(
        result["burn_in_action"].numpy()
        == [
            [0, 1, 2, 3, 4],
            [0, 0, 0, 0, 0],
            [25, 26, 27, 28, 29],
            [0, 0, 0, 0, 0],
        ]
    )

    assert np.all(
        result["obs"].numpy()
        == [
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        ]
    )
    assert np.all(
        result["action"].numpy()
        == [
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44],
        ]
    )


@pytest.fixture
def mock_table():
    class MockTable:
        def _execute_gather(self, count, length, sample_points):
            out = {}
            for key, size in (("obs", 10), ("action", 5)):
                data = torch.arange(count * 5 * size).reshape(count * 5, size)
                storage = torch.zeros(count * length, size)
                counter = 0
                for row, start, end in sample_points:
                    storage_start = counter * length
                    storage_end = counter * length + end - start
                    data_start = row * 5 + start
                    data_end = row * 5 + end
                    storage[storage_start:storage_end, :] = data[data_start:data_end, :]
                    counter += 1

                out[key] = storage

            return out

    return MockTable()


def mock_results(offset):
    rollout_length = 3
    obs_size = 10
    action_size = 5

    obs_offset = offset * obs_size
    action_offset = offset * action_size
    result = dict(
        obs=torch.concat(
            [
                torch.arange(
                    obs_offset, obs_offset + obs_size * rollout_length
                ).reshape(rollout_length, obs_size),
                torch.arange(
                    50 + obs_offset, 50 + obs_offset + obs_size * rollout_length
                ).reshape(rollout_length, obs_size),
            ]
        ),
        action=torch.concat(
            [
                torch.arange(
                    action_offset, action_offset + action_size * rollout_length
                ).reshape(rollout_length, action_size),
                torch.arange(
                    action_offset + 25,
                    action_offset + 25 + action_size * rollout_length,
                ).reshape(rollout_length, action_size),
            ]
        ),
        sample_points=np.array(
            [
                [0, offset, offset + 3],
                [1, offset, offset + 3],
            ]
        ),
    )

    print(result)
    return result


def test_adaptor_start_at_two(mock_table):
    """When the sampling point is at random the burn in lengths should be the actual length of the
    burn in and the burn in data should be the prefix of the regular data.

    """

    adaptor = BurnInSamplerAdaptor(["obs", "action"], 2)
    adaptor._table = mock_table

    result = adaptor(mock_results(2), 2, 3)

    assert np.all(result["burn_in_lengths"].numpy() == [2, 2])
    assert np.all(result["burn_in_masks"].numpy() == [1, 1])
    assert np.all(result["burn_in_obs"].numpy().shape == (4, 10))
    assert np.all(
        result["burn_in_obs"].numpy()
        == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        ]
    )

    assert np.all(
        result["burn_in_action"].numpy()
        == [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
        ]
    )

    assert np.all(
        result["obs"].numpy()
        == [
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        ]
    )
    assert np.all(
        result["action"].numpy()
        == [
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
        ]
    )


def test_adaptor_start_at_two_actions_obs_extra(mock_table):
    """When the sampling point is at random the burn in lengths should be the actual length of the
    burn in and the burn in data should be the prefix of the regular data.

    """
    adaptor = BurnInSamplerAdaptor(["obs", "action"], 2)
    adaptor._table = mock_table

    result = adaptor(mock_results(2), 2, 5)

    assert np.all(result["burn_in_lengths"].numpy() == [2, 2])
    assert np.all(result["burn_in_masks"].numpy() == [1, 1])
    assert np.all(result["burn_in_obs"].numpy().shape == (4, 10))
    assert np.all(
        result["burn_in_obs"].numpy()
        == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        ]
    )

    assert np.all(
        result["burn_in_action"].numpy()
        == [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
        ]
    )

    assert np.all(
        result["obs"].numpy()
        == [
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        ]
    )
    assert np.all(
        result["action"].numpy()
        == [
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
        ]
    )
