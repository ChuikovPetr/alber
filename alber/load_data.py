from typing import Tuple

import h5py
import numpy as np
import pandas as pd


def get_two_columns_of_ob(
    name: str, folder_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(np.array(h5py.File(folder_name)["OB"][name]))
    df.columns = [i for i in range(df.shape[1])]
    df = df[[0, 1]]
    df1 = df[[0]]
    df2 = df[[1]]
    del df
    return df1, df2


def read_order_book(folder_name: str) -> pd.DataFrame:
    time = pd.DataFrame(np.array(h5py.File(folder_name)["OB"]["TS"]))

    bid_price1, bid_price2 = get_two_columns_of_ob("Bid", folder_name)
    ask_price1, ask_price2 = get_two_columns_of_ob("Ask", folder_name)

    bid_size1, bid_size2 = get_two_columns_of_ob("BidV", folder_name)
    ask_size1, ask_size2 = get_two_columns_of_ob("AskV", folder_name)

    res = pd.concat(
        [
            time,
            bid_price1,
            bid_price2,
            ask_price1,
            ask_price2,
            bid_size1,
            bid_size2,
            ask_size1,
            ask_size2,
        ],
        axis=1,
    )

    res.columns = [
        "time",
        "bid_price1",
        "bid_price2",
        "ask_price1",
        "ask_price2",
        "bid_size1",
        "bid_size2",
        "ask_size1",
        "ask_size2",
    ]
    return res


def read_trades(folder_name: str) -> pd.DataFrame:
    time = pd.DataFrame(np.array(h5py.File(folder_name)["Trades"]["TS"]))
    price = pd.DataFrame(np.array(h5py.File(folder_name)["Trades"]["Price"]))
    size = pd.DataFrame(np.array(h5py.File(folder_name)["Trades"]["Amount"]))

    trades = pd.concat([time, price, size], axis=1)
    trades.columns = ["time", "price", "size"]

    sum_w = (
        trades[["time", "size"]].groupby("time").agg([np.sum, "count"]).reset_index()
    )
    trades = pd.merge(trades, sum_w, on=["time"])
    trades.columns = ["time", "price", "size", "sum_size", "order_count"]
    trades["weight_price"] = trades["size"] * trades["price"]
    trades = (
        trades.groupby("time")
        .agg(
            {
                "price": np.max,
                "size": np.max,
                "sum_size": np.max,
                "order_count": np.max,
                "weight_price": np.sum,
            }
        )
        .reset_index()
    )
    trades["price"] = trades["weight_price"] / trades["sum_size"]
    trades = trades[["time", "price", "sum_size", "order_count"]]
    trades = trades.rename(columns={"sum_size": "size"})

    return trades


def read_target(folder_name: str) -> pd.DataFrame:
    time = pd.DataFrame(np.array(h5py.File(folder_name)["Return"]["TS"]))
    target = pd.DataFrame(np.array(h5py.File(folder_name)["Return"]["Res"]))

    res = pd.concat([time, target], axis=1)
    res.columns = ["time", "target"]
    res = res.astype({"time": int})
    res = res.drop_duplicates(["time"]).reset_index(drop=True)

    return res
