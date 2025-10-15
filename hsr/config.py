import os
import pandas as pd

DEFAULT_PATH=os.path.join(os.path.expanduser("~"), "work/data/hsr")
if not os.path.exists(DEFAULT_PATH):
    os.makedirs(DEFAULT_PATH)

FUNDA_PATH=os.path.join(os.path.expanduser("~"), "work/data/data_feed/bbg/fundamentals")

universe = "r3000"
start_date = pd.to_datetime("2015-01-01")
end_date = pd.Timestamp("now")
identifier = "Ticker"
region = "US"
date_col = "as_of_date"