# 🌡 Metrics

Emote can log metrics from two locations: inside the training loop, and outside the training
loop. The base for this is the [`LoggingMixin`](emote.callbacks.logging.LoggingMixin) class in both cases,
adds logging functionality to anything. However, it doesn't do any actual logging.

On the training side, the second part of the puzzle is a LogWriter, for example
[`TensorboardLogger`](emote.callbacks.logging.TensorboardLogger). We also provide a built-in
[`TerminalLogger`](emote.callbacks.logging.TerminalLogger). These accept a list of objects derived from
[`LoggingMixin`](emote.callbacks.logging.LoggingMixin), and will execute the actual writing of values from
the previously of values. This makes implementing log-data-providers easier, as they do not have to
care about *when to write*, only how often they can record data.

```python
logger = SystemLogger()
tensorboard_log_writer = TensorboardLogger([logger], SummaryWriter("/tmp/output_dir"), 2000)
trainer = Trainer([logger, tensorboard_log_writer])
```

Things behave slightly differently on the data-generation side. Our suggested (and only supported
method) is to wrap the memory with a [`LoggingProxyWrapper`](emote.memory.memory.LoggingProxyWrapper). Since all data going into the training loop passes through the memory, and all data has associated metadata, this will capture most metrics.

Our suggestion is that users primarily rely on this mechanism for logging data associated with the
agents, as it will get smoothed across all agents to reduce noise.


```python
env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
table = DictObsMemoryTable(spaces=env.dict_space, maxlen=1000, device="cpu")
table_proxy = MemoryTableProxy(table, 0, True)
table_proxy = LoggingProxyWrapper(table, SummaryWriter("/tmp/output_dir"), 2000)
```
