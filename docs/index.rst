=========
🍒 Emote
=========

**Emote** — **E**\ mbark's **Mo**\ dular **T**\ raining **E**\ ngine — is a flexible framework
for reinforcement learning written at Embark.

Installation
============
Install `Poetry <https://python-poetry.org/>`_ following the instructions on the
Poetry site. Then install the package using ::

   poetry install


Ideas and Philosophy
====================

We wanted a reinforcement learning framework that was modular both in the
sense that we could easily swap the algorithm we used and how data was collected
but also in the sense that the different parts of various algorithms could be reused
to build other algorithms.

.. automodule:: emote


.. toctree::
   :maxdepth: 2
   :caption: Design
   :hidden:

   self

.. toctree::
   :maxdepth: 6
   :caption: API Documentation
   :hidden:

   emote
   memory
   callback

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
