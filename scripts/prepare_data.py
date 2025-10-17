import os
import sys
import pandas as pd
from matterhorn.universe.universe_loader import UniverseLoader
from hsr.config import FUNDA_PATH
from IPython import embed


fundamentals = [
    "IS_SH_FOR_DILUTED_EPS",
    "TOT_COMMON_EQY",
    "SALES_REV_TURN",
    "CF_CASH_FROM_OPER",
    "IS_NET_INC_AVAIL_COM_SHRHLDRS",
    "PFD_EQTY_MINORTY_INTEREST",
    "IS_REGULAR_CASH_DIVIDEND_PER_SH",
    "IS_EPS",
    "BS_LT_BORROW",
    "BS_ST_DEBT",
    "BS_TOT_ASSET",
    "CAPEX_ABSOLUTE_VALUE",
]

def main():
    # universe
    all_tickers = UniverseLoader(data_type="r3000").get()
    all_tickers = [f"{t} US Equity" for t in all_tickers]

    out_path = os.path.join(os.path.expanduser("~"), f"Downloads/bbg/hsr_20250928")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    funda_fn = os.path.join(out_path, "tickers.csv")
    with open(funda_fn, "w") as f:
        f.write(f"\n".join(all_tickers))

    fn = os.path.join(out_path, "fundamentals2.xlsx")  # field name instead of field mnemonic
    sheets = pd.read_excel(fn, sheet_name=None)
    existing_tickers = set()
    for name, df in sheets.items():
        if "US Equity" not in name:
            continue
        ticker = name.split(" ")[0].lower()
        
        out_fn = FUNDA_PATH +f"/{ticker}.csv"
        df.to_csv(out_fn, index=False)
        print(f"wrote {len(df)} rows to {out_fn}")

        existing_tickers.add(f"{ticker.upper()} US Equity")

    all_tickers = list(set(all_tickers) - existing_tickers)
    with open(funda_fn, "w") as f:
        f.write(f"\n".join(all_tickers))
    

if __name__ == "__main__":
    sys.exit(main())
        