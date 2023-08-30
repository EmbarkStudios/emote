import numpy as np

from .forget_me_not import ForgetMeNot


def run_game_helper(game, game_length, num_actions):
    # FIRST STEP: PATTERN
    obs = game.reset()[0]
    assert obs[0] == 1.0, "obs[0] should be 1.0 after reset()"
    assert obs[1] == 0.0, "obs[1] should be 0.0 after reset()"

    # PATTERN STEP
    for _ in range(game_length - 2 - game.get_memory_gap()):
        obs, reward, done, _, _ = game.step(np.zeros(num_actions))
        assert obs[0] == 1.0
        assert obs[1] == 0.0
        assert not done
        assert reward == 0.0

    # NO INFO STEP
    for _ in range(game.get_memory_gap() - 1):
        obs, reward, done, _, _ = game.step(np.zeros(num_actions))
        assert obs[0] == 0.0
        assert obs[1] == 0.0
        assert not done
        assert reward == 0.0

    # MEMORY TRIGGER STEP
    obs, reward, done, _, _ = game.step(np.zeros(num_actions))
    assert obs[0] == 0.0
    assert obs[1] == 1.0
    assert not done
    assert reward == 0.0


def test_forget_me_not_incorrect_pattern():
    num_actions = 5
    game_length = 3
    memory_gap = 1
    game = ForgetMeNot(
        num_actions,
        game_length=game_length,
        min_memory_gap=memory_gap,
        max_memory_gap=memory_gap,
        difficulty=1,
    )

    run_game_helper(game, game_length, num_actions)

    # FINAL STEP
    obs, reward, done, _, _ = game.step(game.get_pattern() / 2.0)
    assert obs[0] == 0.0
    assert obs[1] == 1.0
    assert done
    assert reward < 1.0


def test_forget_me_not_correct_pattern():
    num_actions = 5
    game_length = 3
    memory_gap = 1
    game = ForgetMeNot(
        num_actions,
        game_length=game_length,
        min_memory_gap=memory_gap,
        max_memory_gap=memory_gap,
        difficulty=1,
    )

    run_game_helper(game, game_length, num_actions)

    # FINAL STEP
    obs, reward, done, _, _ = game.step(game.get_pattern())
    assert obs[0] == 0.0
    assert obs[1] == 1.0
    assert done
    assert reward == 1.0


def test_forget_me_not_correct_pattern_long_game():
    num_actions = 5
    game_length = 10
    memory_gap = 1
    game = ForgetMeNot(
        num_actions,
        game_length=game_length,
        min_memory_gap=memory_gap,
        max_memory_gap=memory_gap,
        difficulty=1,
    )

    run_game_helper(game, game_length, num_actions)

    # FINAL STEP
    obs, reward, done, _, _ = game.step(game.get_pattern())
    assert obs[0] == 0.0
    assert obs[1] == 1.0
    assert done
    assert reward == 1.0


def test_forget_me_not_correct_pattern_long_game_large_gap():
    num_actions = 5
    game_length = 10
    memory_gap = 2
    game = ForgetMeNot(
        num_actions,
        game_length=game_length,
        min_memory_gap=memory_gap,
        max_memory_gap=memory_gap,
        difficulty=1,
    )

    run_game_helper(game, game_length, num_actions)

    # FINAL STEP
    obs, reward, done, _, _ = game.step(game.get_pattern())
    assert obs[0] == 0.0
    assert obs[1] == 1.0
    assert done
    assert reward == 1.0


def test_forget_me_not_correct_pattern_long_game_variable_gap():
    num_actions = 5
    game_length = 10
    memory_gap_min = 2
    memory_gap_max = 5
    game = ForgetMeNot(
        num_actions,
        game_length=game_length,
        min_memory_gap=memory_gap_min,
        max_memory_gap=memory_gap_max,
        difficulty=1,
    )

    for _ in range(5):
        run_game_helper(game, game_length, num_actions)

        # FINAL STEP
        obs, reward, done, _, _ = game.step(game.get_pattern())
        assert obs[0] == 0.0
        assert obs[1] == 1.0
        assert done
        assert reward == 1.0


def test_forget_me_not_no_gap():
    num_actions = 5
    game_length = 3
    memory_gap = 0
    game = ForgetMeNot(
        num_actions,
        game_length=game_length,
        min_memory_gap=memory_gap,
        max_memory_gap=memory_gap,
        difficulty=1,
    )

    # FIRST STEP: PATTERN
    obs = game.reset()[0]
    assert obs[0] == 1.0
    assert obs[1] == 0.0

    # MEMORY TRIGGER STEP
    obs, reward, done, _, _ = game.step(np.zeros(num_actions))
    assert obs[0] == 1.0
    assert obs[1] == 1.0
    assert not done
    assert reward == 0.0

    # FINAL STEP
    obs, reward, done, _, _ = game.step(game.get_pattern())
    assert done
