import numpy as np
import pytest

from emote.memory.storage import NextNElementWrapper


@pytest.fixture
def storage() -> np.ndarray:
    return np.arange(32).reshape((2, -1))


@pytest.mark.parametrize(
    ("batch_dim", "n"), ((0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4))
)
def test_next_n_element_single(batch_dim, n, storage):
    wrapper = NextNElementWrapper.with_n(n)(storage, (1,), np.float32)[batch_dim]

    next_0 = wrapper[0]
    next_1 = wrapper[1]
    next_5 = wrapper[5]

    assert next_0 == storage[batch_dim][n]
    assert next_1 == storage[batch_dim][1 + n]
    assert next_5 == storage[batch_dim][5 + n]


@pytest.mark.parametrize(
    ("batch_dim", "n"), ((0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4))
)
def test_next_n_element_slice(batch_dim, n, storage):
    wrapper = NextNElementWrapper.with_n(n)(storage, (1,), np.float32)[batch_dim]

    next_0_to_2 = wrapper[0:2]
    next_1_to_4 = wrapper[1:4]
    next_2_to_5_skip_2 = wrapper[2:5:2]

    assert np.all(next_0_to_2 == storage[batch_dim][n : (n + 2)])
    assert np.all(next_1_to_4 == storage[batch_dim][(1 + n) : (4 + n)])
    assert np.all(next_2_to_5_skip_2 == storage[batch_dim][(2 + n) : (5 + n) : 2])


@pytest.mark.parametrize(("batch_dim", "n"), ((0, 1), (0, 2), (1, 1), (1, 2)))
def test_next_n_element_tuple(batch_dim, n, storage):
    storage = np.reshape(storage, (2, 4, 4))

    wrapper = NextNElementWrapper.with_n(n)(storage, (4, 4), np.float32)[batch_dim]

    next_0_0 = wrapper[(0, 0)]
    next_1_0 = wrapper[(1, 0)]
    next_1_1 = wrapper[(1, 1)]

    assert np.all(next_0_0 == storage[batch_dim][(n, n)])
    assert np.all(next_1_0 == storage[batch_dim][(1 + n, n)])
    assert np.all(next_1_1 == storage[batch_dim][(1 + n, 1 + n)])


@pytest.mark.parametrize(("batch_dim", "n"), ((0, 2), (1, 2)))
def test_next_n_element_access_single_out_of_bounds_raises(batch_dim, n, storage):
    wrapper = NextNElementWrapper.with_n(n)(storage, (1,), np.float32)[batch_dim]
    array_len = storage.shape[1]

    with pytest.raises(IndexError):
        wrapper[array_len - 2]
    with pytest.raises(IndexError):
        wrapper[array_len - 1]
    with pytest.raises(IndexError):
        wrapper[array_len]
    with pytest.raises(IndexError):
        wrapper[array_len + 1]


@pytest.mark.parametrize(("batch_dim", "n"), ((0, 2), (1, 2)))
def test_next_n_element_access_slice_out_of_bounds_raises(batch_dim, n, storage):
    wrapper = NextNElementWrapper.with_n(n)(storage, (1,), np.float32)[batch_dim]
    array_len = storage.shape[1]

    with pytest.raises(IndexError):
        wrapper[array_len - 2 : array_len]
    with pytest.raises(IndexError):
        wrapper[array_len + 2 : array_len + 4]


@pytest.mark.parametrize(("batch_dim", "n"), ((0, 2), (1, 2)))
def test_next_n_element_access_tuple_out_of_bounds_raises(batch_dim, n, storage):
    storage = np.reshape(storage, (2, 4, 4))

    wrapper = NextNElementWrapper.with_n(n)(storage, (1,), np.float32)[batch_dim]

    with pytest.raises(IndexError):
        wrapper[(3, 3)]

    with pytest.raises(IndexError):
        wrapper[(4, 4)]
