# Shoggoth

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