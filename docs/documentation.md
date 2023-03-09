# 📚 Documentation

To write documentation for *emote* we support both classic *ReStructured Text* (`.rst`) and modern `Markdown` (`.md`) files. These can also reference each other, though for ease of use a tree should maintain the same type.

To include RST text into Markdown code, use the following pattern:


    ```{eval-rst}
    .. include:: snippets/include-rst.rst
	```

That is to say, a code-block with the `eval-rst` directive and then a verbatim include of the markdown contents. The opposite, including Markdown in ReST can be achieved with this recipe:

    .. include:: include.md
       :parser: myst_parser.sphinx_

See the [Myst documentation](https://myst-parser.readthedocs.io/en/latest/faq/index.html) for more recipes and directives.

## Helpful commands

* To build the docs: `pdm run docs`
* To view the docs in your browser: `pdm run docs-serve`
