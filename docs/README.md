# TorchTPU Documentation

This directory contains the TorchTPU documentation.

## Build locally *(for generating the static HTML files)*

To build the documentation locally, follow these steps:

1. From the root of the repository, install the documentation dependencies listed in `docs/requirements.txt`:
   ```bash
   python -m pip install -r docs/requirements.txt
   ```

2. From the root of the repo, run the following command:

```bash
mkdocs build
```

3. The generated HTML files will be located in the `site` directory. Open
`index.html` in your browser to view the documentation.

## Serve locally *(recommended for local previews)*

To serve the documentation locally, follow these steps:

1. Install the documentation dependencies from the root of the repository:
```bash
python -m pip install -r docs/requirements.txt
```
2. From the root of the repository, run the following command:
```bash
mkdocs serve
```
3. Open your browser and navigate to `http://127.0.0.1:8000/` to view the
documentation locally.

## Sources for the documentation

The documentation is built from the [Marimo](https://marimo.io/) sources located
in the `docs/source/` directory. These files are Marimo `.py` notebooks. They
serve as the single source of truth for TorchTPU documentation and tutorials.
Jupyter notebooks (`.ipynb`) and Markdown exports are generated from these
files.

The `docs/markdown/` directory contains the generated Markdown exports from the
Marimo sources. These Markdown files are used by MkDocs to build the HTML
documentation.
