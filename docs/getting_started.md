# 🔥 Getting Started

In the `/experiments` folder, example runs can be found for different Gymnasium environments.

For example, you can run the cartpole example using DQN with the following command:

```python
pdm run python experiments/gym/train_dqn_cartpole.py
```

![Alt Text](cart_pole.gif)

This comes with a lot of predefined arguments, such as the learning rate, the amount of hidden layers, the batch size, etc. You can find all the arguments in the `experiments/gym/train_dqn_cartpole.py` file.

## 📊 Tensorboard

To visualize the training process, you can use Tensorboard. To do so, run the following command:

```bash
pdm run tensorboard --logdir ./mllogs
```

This will start a Tensorboard server on `localhost:6006`. You can now open your browser and go to `localhost:6006` to see the training process where you can see the rewards over time, the loss over time, etc.

![Alt Text](tensorboard.png)
