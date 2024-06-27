# 📚 Documentation

To write documentation for *emote* we use [mdBook](https://rust-lang.github.io/mdBook/) written in `Markdown` (`.md`) files. These can reference each other, and will be built into a book like HTML bundle.

See the [mdBook markdown docs](https://rust-lang.github.io/mdBook/format/markdown.html) for details about syntax and feature support.

## Helpful commands

* To build the docs: `pants package docs:docs`
* To view the docs in your browser: `pants run docs:serve`
