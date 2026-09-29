"""Download Optiver "Trading at the Close" train.csv and convert it to Parquet.

Needs a Kaggle login (`.venv/bin/kaggle auth login`) and the competition rules
accepted at https://www.kaggle.com/competitions/optiver-trading-at-the-close/data.
The data is never committed (Kaggle competition rules forbid redistribution).
"""

from markout.auction.load import main

if __name__ == "__main__":
    main()
