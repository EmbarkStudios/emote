=========
🍒 Emote
=========

**Emote** — **E**\ mbark's **Mo**\ dular **T**\ raining **E**\ ngine — is a flexible framework
for reinforcement learning written at Embark.

Installation
============

Install `PDM <https://pdm.fming.dev/latest/#installation>`_ following the instructions on the
PDM site. Then install the package using ::

   pdm install


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
   installing-torch
   coding-standard
   📚 Editing documentation <documentation.md>
   🌡 Metrics <metrics.md>
   🚀 Getting Started <getting_started.md>

.. toctree::
   :maxdepth: 6
   :caption: API Documentation
   :hidden:

   emote
   memory
   callback
   callbacks

.. toctree::
   :caption: Extras
   :hidden:
   :glob:

   Undocumented code <coverage.rst>
   Architecture Desicision Records <adr/doc.md>

..
   .. include:: adr/doc.md
   .. include:: documentation.md
   .. include:: metrics.md
   .. include:: getting_started.md

	  :parser: myst_parser.sphinx_

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
