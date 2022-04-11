=========
🍒 Emote
=========

**Emote** — **E**\ mbark's **Mo**\ dular **T**\ raining **E**\ ngine — is a flexible framework
for reinforcement learning written at Embark.

Installation
============

.. warning:: You'll need to use a pre-release Poetry 1.2 version, e.g. 1.2.0a1 or later. Older versions will crash while installing our dependencies.

Install `Poetry <https://python-poetry.org/>`_ following the instructions on the
Poetry site. Then install the package using ::

   poetry install

MacOS instructions
------------------

On MacOS specifically you'll need a working C/C++ compiler as well as `swig <https://www.swig.org/>`_. The easiest way to install it is via homebrew: ::

  brew install swig


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
   coding-standard

.. toctree::
   :maxdepth: 6
   :caption: API Documentation
   :hidden:

   emote
   memory
   callback

.. toctree::
   :caption: Extras
   :hidden:

   Undocumented code <coverage.rst>

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
