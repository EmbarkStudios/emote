<!-- Allow this file to not have a first line heading -->
<!-- markdownlint-disable-file MD041 -->

<!-- inline html -->
<!-- markdownlint-disable-file MD033 -->

<div align="center">

# 🍒 Shoggoth

> It was a terrible, indescribable thing vaster than any subway train — a
> shapeless congeries of protoplasmic bubbles, faintly self-luminous, and
> with myriads of temporary eyes forming and un-forming as pustules of greenish
> light all over the tunnel-filling front that bore down upon us, crushing the
> frantic penguins and slithering over the glistening floor that it and its
> kind had swept so evilly free of all litter.

Shoggoth is a flexible framework for reinforcement learning written at Embark.
It is not the final title for the repository but I needed a placeholder.
It is my humble hope that whatever code we produce will be more structured than the
average shoggoth.
    
[![Embark](https://img.shields.io/badge/embark-open%20source-blueviolet.svg)](https://embark.dev)
[![Embark](https://img.shields.io/badge/discord-ark-%237289da.svg?logo=discord)](https://discord.gg/dAuKfZS)
[![Documentation Status](https://readthedocs.org/projects/shoggoth/badge/?version=latest)](http://shoggoth.readthedocs.io/?badge=latest)
[![PyPI version fury.io](https://badge.fury.io/py/shoggoth.svg)](https://pypi.python.org/pypi/shoggoth/)
[![Build status](https://github.com/EmbarkStudios/shoggoth/workflows/CI/badge.svg)](https://github.com/EmbarkStudios/shoggoth/actions)
</div>


## Installation

Install [Poetry](https://python-poetry.org/) following the instructions on the
Poetry site. Then install the package using

```bash
poetry install
```

### Installation on Windows

Some of the development dependencies require a working compiler. Install 
[MSYS2](https://www.msys2.org/), then install `mingw-64` and `swig` using `pacman`.
After that you will probably have to invoke `poetry install` from inside the
`MSYS2 MINGW 64` shell.


## Contribution

[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4-ff69b4.svg)](../main/CODE_OF_CONDUCT.md)

We welcome community contributions to this project.

Please read our [Contributor Guide](CONTRIBUTING.md) for more information on how to get started.
Please also read our [Contributor Terms](CONTRIBUTING.md#contributor-terms) before you make any contributions.

Any contribution intentionally submitted for inclusion in an Embark Studios project, shall comply with the Rust standard licensing model (MIT OR Apache 2.0) and therefore be dual licensed as described below, without any additional terms or conditions:

### License

This contribution is dual licensed under EITHER OF

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

For clarity, "your" refers to Embark or any other licensee/user of the contribution.
