🔥 Installing Torch
===================

Pytorch supports multiple variants of hardware and compute
backends. Due to how Python versioning works and how Pytorch publishes
their packages; it is impossible to use all of these as dependencies,
optional or not. We still want to make it easy and quick to install
this package, and develop it.

As a starting step, we offer the extra group `torch`. As the name
implies, this'll use our pinned version of torch with CPU sort and the
default CUDA support. This is quick and easy to install, and works
well for testing and development.

GPU support
-----------

Depending on how you use ``emote``, you'll need to approach GPU support slightly differently.

Using the emote repository and PDM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For those wanting GPU support; we have a plugin for PDM to help install these variants while maintaining compatibility
and the lockfile securities. This plugin is exposed as ``pdm torch`` after installing the repository.

To install a specific backend API for torch; use the command ``pdm torch install {API}``. We have enabled ``cpu``, ``cu116``,
and ``rocm5.0`` by default. If you're using the repository and something is missing that you need, feel free to
add it and PR it back to us. Any backend selected here has to be available from a PEP503 or PEP621 page hosted by
PyTorch.

When installing from PyPi
^^^^^^^^^^^^^^^^^^^^^^^^^

Our suggestion is to avoid mixing package managers and Python Interpreters. We'd suggest following the method of the
plugin and installing torch from PyTorch's PEP503 index. For example; for CUDA 11.1 you can use the command ``pip
install -i https://download.pytorch.org/whl/cu116/ torch==$TORCH_VERSION``. There's other pages for other APIs. There's
also a generic `repository <https://download.pytorch.org/whl/>`_ which provides a combination of APIs, such as pure CPU
and rocm.
