emote.callback
==============

.. currentmodule:: emote.callback
.. automodule:: emote.callback

In this module you'll find the callback framework used by Emote. Those
who have used FastAI before will recognize it, as it's heavily
inspired by that system - but adapted for RL and our use-cases.

The `Callback` interface
########################

The callback is the core interface used to hook into the Emote framework. You can think of these as events - when the training loop starts, we'll invoke `begin_training` on all callback objects. Then we'll start a new cycle, and call :meth:`Callback.begin_cycle` for those that need it.

All in all, the flow of callbacks is like this:

.. graphviz::

    digraph foo {
      rankdir=LR;
      node [shape=rectangle,style="rounded"]

      newrank=true;
      { rank=same;  begin_cycle;  begin_batch; }
      { rank=same;  end_batch; end_cycle; }

	  restore_state;
      begin_training;
      subgraph cluster_cycle {
        label = "while cycles left"
        begin_cycle;

        subgraph cluster_batch {
          label = "while batches left";
          begin_batch;
          backward;
          end_batch;
        }
        end_cycle;
      }
	  end_training;

      restore_state -> begin_training -> begin_cycle -> begin_batch -> backward -> end_batch;

      end_batch -> begin_batch [constraint=no];
      end_cycle -> begin_cycle [constraint=no];

      end_batch -> end_cycle [style=dashed]
      end_cycle -> end_training [style=dashed]
   }

.. autoclass:: Callback
   :members:
   :member-order: bysource
