# hornli-hsr

Open risk/alpha infrastructure for quants.

## Overview

hornli-hsr provides composable building blocks for quantitative researchers to design, backtest, and deploy systematic strategies. It offers utilities for data ingestion, portfolio construction, risk analysis, and result visualization so that teams can iterate quickly while retaining reproducibility.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- GNU Make
- Recommended: a virtual environment manager such as `venv` or `conda`

### Installation

```bash
# create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# install hornli-hsr
pip install -e .
```

### Quickstart

Run the demo workflow to verify the installation:

```bash
make demo
```

This will download sample market data, execute a toy strategy, and generate a simple performance report in the `reports/` directory.

## Project Structure

- `hsr/`: Core library code including data pipelines, analytics modules, and strategy interfaces.
- `scripts/`: Command-line utilities and helper scripts for maintenance and batch tasks.
- `Makefile`: Common automation targets for linting, testing, and running demos.
- `setup.py`: Editable installation entry point.

## Development

We welcome contributions! To run the codebase locally:

```bash
pip install -r requirements-dev.txt
make lint test
```

Before submitting changes, ensure that the full test suite passes and that any new features are documented in this README or the project documentation.

## License

hornli-hsr is released under the MIT License. See `LICENSE` for details.
