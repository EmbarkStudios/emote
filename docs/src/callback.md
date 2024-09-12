# Callback system


In this module you'll find the callback framework used by Emote. Those
who have used FastAI before will recognize it, as it's heavily
inspired by that system - but adapted for RL and our use-cases.

## The `Callback` interface

The callback is the core interface used to hook into the Emote framework. You can think of these as events - when the training loop starts, we'll invoke `begin_training` on all callback objects. Then we'll start a new cycle, and call :meth:`Callback.begin_cycle` for those that need it.

All in all, the flow of callbacks is like this:

![Dot Graph of Callback flow](./callback.png)
