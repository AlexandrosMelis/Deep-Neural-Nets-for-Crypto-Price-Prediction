from os.path import join
import pandas as pd
from config import PathConfig


def read_raw_data(coin_name: str, path: str = PathConfig.data_4h_interval_path.value, include_rsi: bool = False, include_macd: bool = False) -> pd.DataFrame:
    # available coins: ADA, BNB, BTC, DASH, ETH, LINK, LTC, XRP
    # conc: "rsi", "macd", "macd_rsi"

    data_path = join(path, coin_name ,f"{coin_name.lower()}_2019.csv")
    data_df = pd.read_csv(data_path, index_col=False)

    if include_rsi:
        rsi_path = join(path, coin_name, f"{coin_name.lower()}_4h_rsi.csv")
        rsi_df = pd.read_csv(rsi_path, index_col=False)
        rsi_df = rsi_df.drop('Time', axis=1)
        rsi_df = rsi_df.rename(columns={"0": "rsi"})
        data_df = pd.concat([data_df, rsi_df], axis="columns")
    if include_macd:
        macd_path = join(path, coin_name, f"{coin_name.lower()}_4h_macd.csv")
        macd_df = pd.read_csv(macd_path, index_col=False)
        macd_df = macd_df.drop('Time', axis=1)
        macd_df = macd_df.rename(columns={"0": "macd"})
        data_df = pd.concat([data_df, macd_df], axis="columns")
    
    data_df.dropna(inplace=True)
    data_df.columns = [col.lower() for col in data_df.columns]
    return data_df


def read_base_data(path: str = PathConfig.base_data_path.value) -> dict:
    data = {}
    files = ['train', 'valid', 'test']
    for file in files:
        data[file] = pd.read_csv(join(path, f"{file}.csv"), index_col='time')
    return data