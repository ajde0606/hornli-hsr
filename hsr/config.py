import os
from dotenv import load_dotenv
load_dotenv()                      # reads .env in CWD
import pandas as pd

DEFAULT_PATH=os.environ['HSR_PATH']

if not os.path.exists(DEFAULT_PATH):
    os.makedirs(DEFAULT_PATH)

if not os.path.exists(DEFAULT_PATH+"/input"):
    os.makedirs(DEFAULT_PATH+"/input")

if not os.path.exists(DEFAULT_PATH+"/intermediate"):
    os.makedirs(DEFAULT_PATH+"/intermediate")

if not os.path.exists(DEFAULT_PATH+"/output"):
    os.makedirs(DEFAULT_PATH+"/output")

FUNDA_PATH = os.path.join(DEFAULT_PATH, "input/fundamentals")
if not os.path.exists(FUNDA_PATH):
    os.makedirs(FUNDA_PATH)

universe = "r3000"
start_date = pd.to_datetime("2015-01-01")
end_date = pd.Timestamp("now")
identifier = "Ticker"
region = "US"
date_col = "as_of_date"