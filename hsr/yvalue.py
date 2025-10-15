import os
import numpy as np
import pandas as pd
from hsr.config import *
from matterhorn.data_loader.price_loader import PriceLoader


window = 21
loader = PriceLoader("US",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    'r3000',
                    'Ticker'
                    )
data = loader.load("price")
adj_close = data.pivot(index="as_of_date", columns="Ticker", values="adj_close")
simple_ret = np.log(adj_close / adj_close.shift(window)) / window

out_fn = os.path.join(DEFAULT_PATH, "simple_return.parquet")
simple_ret.to_parquet(out_fn)

print(f"Saved simple return to {out_fn}")