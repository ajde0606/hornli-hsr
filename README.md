# hornli-hsr

Open risk model infrastructure for quants.

## Overview

hornli-hsr provides a free open-source risk model for quantitative researchers. Along with a fully documented methodology, we're releasing the full dataset for free, including descriptors, loadings, factor returns, factor covariance, specific risk and specific return.
The methodology can be found at https://hornliquant.com/api/risk-model/whitepaper
Please see https://hornliquant.com/risk-model for more details

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

## Project Structure

- `hsr/`: Core library code including descriptors calculation, loadings calculation, cross-sectional regression framework, covariance estimation and analytics modules
- `setup.py`: Editable installation entry point.
- `example.ipynb`: Explains how to download data and run all risk model calculation


## Quick Start

```bash
echo "HSR_PATH=(Enter your data path)" >> .env
```
Then, try example.ipynb

## License
hornli-hsr is released under the MIT License. See `LICENSE` for details.
