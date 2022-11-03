from typing import List

import numpy as np
import pandas as pd


# +
# Function to calculate first WAP
def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


# Function to calculate second WAP
def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series: pd.Series) -> np.ndarray:
    return np.log(series).diff()


def calc_mean_price(df: pd.DataFrame) -> pd.Series:
    mp = (df["bid_price1"] + df["ask_price1"]) / 2
    return mp


# Function to preprocess book data (for each stock id)
def book_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate Wap
    df["wap1"] = calc_wap1(df)
    df["wap2"] = calc_wap2(df)
    df["mean_price"] = calc_mean_price(df)
    # Calculate log returns
    df["log_return1"] = df[["wap1"]].apply(log_return)
    df["log_return2"] = df[["wap2"]].apply(log_return)
    df["log_return_mean_price"] = df[["mean_price"]].apply(log_return)
    # Calculate wap balance
    df["wap_balance"] = df["wap1"] / df["wap2"]
    # Calculate spread
    df["price_spread"] = (df["ask_price1"] - df["bid_price1"]) / (
        (df["ask_price1"] + df["bid_price1"]) / 2
    )

    df["bid_ask_spread1"] = df["ask_price1"] / df["bid_price1"] - 1
    df["bid_ask_spread2"] = df["ask_price2"] / df["bid_price2"] - 1
    df["bid_spread"] = df["bid_price1"] / df["bid_price2"] - 1
    df["ask_spread"] = df["ask_price1"] / df["ask_price2"] - 1

    df["volume_ask"] = df["ask_size1"] + df["ask_size2"]
    df["volume_bid"] = df["bid_size1"] + df["bid_size2"]
    df["total_volume"] = df["volume_ask"] + df["volume_bid"]
    df["volume_imbalance"] = df["volume_ask"] - df["volume_bid"]
    df["abs_volume_imbalance"] = abs(df["volume_ask"] - df["volume_bid"])

    df["rel_volume_ask_bid"] = df["volume_ask"] / df["volume_bid"] - 1
    df["rel_volume_ask_bid1"] = df["ask_size1"] / df["bid_size1"] - 1
    df["rel_volume_ask_bid2"] = df["ask_size2"] / df["bid_size2"] - 1
    df["rel_volume_ask"] = df["ask_size1"] / df["ask_size2"] - 1
    df["rel_volume_bid"] = df["bid_size1"] / df["bid_size2"] - 1

    df["bid_ask_w_spread1"] = (df["ask_price1"] * df["ask_size1"]) / (
        df["bid_price1"] * df["bid_size1"]
    ) - 1
    df["bid_ask_w_spread2"] = (df["ask_price2"] * df["ask_size2"]) / (
        df["bid_price2"] * df["bid_size2"]
    ) - 1
    df["bid_w_spread"] = (df["bid_price1"] * df["bid_size1"]) / (
        df["bid_price2"] * df["bid_size2"]
    ) - 1
    df["bid_w_ask"] = (df["ask_price1"] * df["ask_size1"]) / (
        df["ask_price2"] * df["ask_size2"]
    ) - 1

    return df.drop(
        [
            "bid_price1",
            "bid_price2",
            "ask_price1",
            "ask_price2",
            "bid_size1",
            "bid_size2",
            "ask_size1",
            "ask_size2",
        ],
        axis=1,
    )


# +
def get_zscore(
    df: pd.DataFrame, span: int, field: str, group_fields: List[str]
) -> pd.DataFrame:
    res_i_list = [i for i in range(len(group_fields))]

    df["mean"] = (
        df.groupby(group_fields)[field]
        .rolling(span, min_periods=1)
        .apply(lambda x: np.mean(x))
        .reset_index(res_i_list, drop=True)
    )

    df["std"] = (
        df.groupby(group_fields)[field]
        .rolling(span, min_periods=1)
        .apply(lambda x: np.std(x) + 0.0000001)
        .reset_index(res_i_list, drop=True)
    )

    df[f"{field}_z"] = df.apply(lambda x: (x[field] - x["mean"]) / x["std"], axis=1)

    return df.drop(["mean", "std"], axis=1)


def get_grid(spans_list: List) -> np.ndarray:
    curr_sl = spans_list
    for span in spans_list:
        curr_sl = sorted(list(set(curr_sl) - set([span])))
        new = np.array(np.meshgrid([span], curr_sl)).T.reshape(-1, 2)
        try:
            res = np.concatenate([res, new])
        except:
            res = new
    return res


def get_trend_features(
    df: pd.DataFrame,
    field: str,
    group_fields: List[str],
    spans: List[int] = [1, 5, 10, 20, 40, 80],
) -> pd.DataFrame:
    res_i_list = [i for i in range(len(group_fields))]
    for span in spans:
        df[f"ma_{span}"] = (
            df.groupby(group_fields)[field]
            .rolling(span)
            .apply(lambda x: np.mean(x))
            .reset_index(res_i_list, drop=True)
        )

    for n_1, n_2 in np.array(get_grid(spans), dtype=int):
        df[f"rel_{field}_{n_1}_{n_2}"] = df.apply(
            lambda x: x[f"ma_{n_1}"] / x[f"ma_{n_2}"], axis=1
        )

    return df.drop([f"ma_{span}" for span in spans], axis=1)


def get_stoch(
    df: pd.DataFrame, span: int, field: str, group_fields: List[str]
) -> pd.DataFrame:
    res_i_list = [i for i in range(len(group_fields))]

    df["MIN"] = (
        df.groupby(group_fields)[field]
        .rolling(span[0])
        .apply(lambda x: np.min(x))
        .reset_index(res_i_list, drop=True)
    )

    df["MAX"] = (
        df.groupby(group_fields)[field]
        .rolling(span[0])
        .apply(lambda x: np.max(x))
        .reset_index(res_i_list, drop=True)
    )

    df["stoch"] = df.apply(
        lambda x: 100 * (x[field] - x["MIN"]) / (0.0000001 + x["MAX"] - x["MIN"]),
        axis=1,
    )

    df[f"stoch_k_{field}_{span[0]}_{span[1]}"] = (
        df.groupby(group_fields)["stoch"]
        .rolling(span[1])
        .apply(lambda x: np.mean(x))
        .reset_index(res_i_list, drop=True)
    )

    df[f"stoch_d_{field}_{span[0]}_{span[2]}"] = (
        df.groupby(group_fields)[f"stoch_k_{field}_{span[0]}_{span[1]}"]
        .rolling(span[2])
        .apply(lambda x: np.mean(x))
        .reset_index(res_i_list, drop=True)
    )

    df[f"rel_stoch_{field}_{span[0]}_{span[1]}_{span[2]}"] = df.apply(
        lambda x: x[f"stoch_k_{field}_{span[0]}_{span[1]}"]
        / (x[f"stoch_d_{field}_{span[0]}_{span[2]}"] + 0.0001),
        axis=1,
    )

    return df.drop(["MIN", "MAX", "stoch"], axis=1)


def get_features_zscore(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
    df[f"high_{period}"] = (
        df.groupby(["id"])["price"]
        .rolling(period)
        .apply(lambda x: np.max(x))
        .reset_index([0], drop=True)
    )

    df[f"low_{period}"] = (
        df.groupby(["id"])["price"]
        .rolling(period)
        .apply(lambda x: np.min(x))
        .reset_index([0], drop=True)
    )

    df[f"open_{period}"] = df.groupby("id")["price"].shift(period - 1)

    df["Ret"] = (df["price"] - df[f"open_{period}"]) / df[f"open_{period}"]
    df["Sprd"] = (df[f"high_{period}"] - df[f"low_{period}"]) / df[f"low_{period}"]
    df["Sprd_Up"] = (df[f"high_{period}"] - df["price"]) / df["price"]
    df["Sprd_Down"] = (df["price"] - df[f"low_{period}"]) / df[f"low_{period}"]

    df = df.drop([f"high_{period}", f"low_{period}", f"open_{period}"], axis=1)

    for name in ["Ret", "Sprd", "Sprd_Up", "Sprd_Down"]:
        print(name)
        df = get_zscore(df, 40, name, ["id"])
        df = df.drop([name], axis=1)
    return df


def get_features_ma(df: pd.DataFrame) -> pd.DataFrame:
    df["Money"] = df["price"] * df["size"]
    for name in ["price", "size", "order_count", "Money"]:
        print(name)
        df = get_trend_features(
            df,
            name,
            ["id"],
        )
    return df


def get_features_stoch(df: pd.DataFrame) -> pd.DataFrame:
    df = get_stoch(df, [14, 1, 3], "price", ["id"])
    df = get_stoch(df, [21, 1, 3], "price", ["id"])
    df = get_stoch(df, [42, 1, 3], "price", ["id"])

    df = get_stoch(df, [14, 1, 3], "size", ["id"])
    df = get_stoch(df, [21, 1, 3], "size", ["id"])
    df = get_stoch(df, [42, 1, 3], "size", ["id"])
    return df


# -


def retime_trades(trades: pd.DataFrame, ob: pd.DataFrame) -> pd.DataFrame:
    ob["id"] = -1
    ob = ob[["id", "time"]]

    trades = pd.concat([trades, ob])
    trades = trades.sort_values(by=["time", "id"])
    return trades.ffill(axis=0).query("id == -1")


def decrease_mem_consuming(
    features: pd.DataFrame, excluding_fields: List[str] = ["id", "date"]
) -> pd.DataFrame:

    new_types = {}
    for name in list(features.columns):
        if name in excluding_fields:
            continue
        if features[name].dtype == "float64":
            new_types[name] = "float32"
        elif features[name].dtype == "int64":
            new_types[name] = "int32"

    features = features.astype(new_types)
    return features
