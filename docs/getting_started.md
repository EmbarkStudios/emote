# 🔥 Getting Started

In the `/experiments` folder, example runs can be found for different Gymnasium environments.

For example, you can run the cartpole example using DQN with the following command:

```python
pdm run python experiments/train_dqn_cartpole.py
```

![Alt Text](cart_pole.gif)

This comes with a lot of predefined arguments, such as the learning rate, the amount of hidden layers, the batch size, etc. You can find all the arguments in the `experiments/train_dqn_cartpole.py` file.

## 📊 Tensorboard

To visualize the training process, you can use Tensorboard. To do so, run the following command:

```bash
pdm run tensorboard --logdir ./mllogs
```

This will start a Tensorboard server on `localhost:6006`. You can now open your browser and go to `localhost:6006` to see the training process where you can see the rewards over time, the loss over time, etc.

![Alt Text](tensorboard.png)

## Command-line Arguments

### `--name`

- **Type**: `str`
- **Default**: `cartpole`
- Specifies the name of the environment.

### `--log-dir`

- **Type**: `str`
- **Default**: `./mllogs/emote/cartpole`
- Directory where logs will be stored.

### `--num-envs`

- **Type**: `int`
- **Default**: `4`
- Number of environments to run in parallel.

### `--rollout-length`

- **Type**: `int`
- **Default**: `1`
- The length of each rollout. Refers to the number of steps or time-steps taken during a simulated trajectory or rollout when estimating the expected return of a policy.

### `--batch-size`

- **Type**: `int`
- **Default**: `128`
- Size of each training batch.

### `--hidden-dims`

- **Type**: `list`
- **Default**: `[128, 128]`
- Dimensions of hidden layers in the neural network.

### `--lr`

- **Type**: `float`
- **Default**: `1e-3`
- The learning rate. Helps in adjusting the model weights.

### `--device`

- **Type**: `str`
- **Default**: `cpu`
- Device to run the model on, e.g., `cpu`, `mps` or `gpu`.

### `--bp-steps`

- **Type**: `int`
- **Default**: `50,000`
- Number of backpropagation steps until the training run is finished.

### `--memory-size`

- **Type**: `int`
- **Default**: `50,000`
- Size of the replay buffer. More complex environments will require a larger replay buffer, as they require more data to learn from. Given that cartpole is a simple environment, a replay buffer of size 50,000 is sufficient.

### `--export-memory`

- **Type**: Flag
- **Default**: `False`
- If set, the replay buffer is exported.
