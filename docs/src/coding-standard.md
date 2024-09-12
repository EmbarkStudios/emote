# 📄 Coding standard

In emote we strive to maintain a consistent style, both visually and
implementation-wise. In order to achieve this we rely on tools to
check and validate our code as we work, and we require that all those
tools are used for CI to pass.

To have a smooth developer experience, we suggest you integrate these
with your editor. We'll provide some example configurations below; and
we welcome contributions to these pages. However, we strive to avoid
*committing* editor configurations to the repository, as that'll more
easily lead to mismatch between different editors - the description
below is authoritative, not any specific editor configuration.

We also require that all commits are made using LF-only line endings. Windows users will need to configure using the below command, or set up their editor appropriately. This helps keep emote platform-generic, and reduces risk for spurious diffs or tools misbehaving.

```shell
git config --global core.autocrlf true
```

## Tools
To run the tools mentioned below on the whole repo, the easiest way is with
```shell
pants lint ::
```


### black


[`Black`](https://github.com/psf/black>) is an auto-formatter for Python,
which mostly matches the PEP8 rules. We use black because it doesn't
support a lot of configuration, and will format for you - instead of
just complaining. We do allow overrides to these styles, nor do we
allow disabling of formatting anywhere.

### isort


[`isort`](https://github.com/PyCQA/isort) is another formatting tool,
but deals only with sorting imports. Isort is configured to be
consistent with Black from within `pyproject.toml`.


## Example configurations


### emacs


```lisp
(use-package python-black
    :demand t
    :after python
    :hook (python-mode . python-black-on-save-mode-enable-dwim))

(use-package python-isort
    :demand t
    :after python
    :hook (python-mode . python-isort-on-save-mode))
```
