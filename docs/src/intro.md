
# 🍒 Emote


**Emote** — **E**\ mbark's **Mo**\ dular **T**\ raining **E**\ ngine — is a flexible framework
for reinforcement learning written at Embark.

## Installation


For package managemend and environment handling we use `pants`. Install it from [pants](https://v1.pantsbuild.org/install.html). After `pants` is set up, verify that it is setup by running

```shell
pants tailor ::
```

## Ideas and Philosophy

We wanted a reinforcement learning framework that was modular both in the
sense that we could easily swap the algorithm we used and how data was collected
but also in the sense that the different parts of various algorithms could be reused
to build other algorithms.
