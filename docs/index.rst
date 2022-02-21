========
Shoggoth
========

    *It was a terrible, indescribable thing vaster than any subway train — a
    shapeless congeries of protoplasmic bubbles, faintly self-luminous, and
    with myriads of temporary eyes forming and un-forming as pustules of greenish
    light all over the tunnel-filling front that bore down upon us, crushing the
    frantic penguins and slithering over the glistening floor that it and its
    kind had swept so evilly free of all litter.*

    --- H. P. Lovecraft, *At the Mountains of Madness*

Shoggoth is a flexible framework for reinforcement learning written at Embark.
It is not the final title for the repository but I needed a placeholder.
It is my humble hope that whatever code we produce will be more structured than the
average shoggoth.

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: shoggoth
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
