# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path

def create_oot(
    vitrine: pd.DataFrame,
    count_time_el_in_test: int,
) -> pd.DataFrame:
    vitrine = vitrine.sort_values(by=["time"]).reset_index(drop=True)
    times = sorted(vitrine.time.unique())

    test_times = times[-count_time_el_in_test:]
   
    vitrine.loc[:, "segment"] = "train"
    vitrine.loc[vitrine.query(f"time == {test_times}").index, "segment"] = "test"

    print(
        f"Splitting vitrine into train/test:\n"
        + f"train_min_time == {vitrine[vitrine['segment'] == 'train'].time.min()}, "
        + f"train_max_time == {vitrine[vitrine['segment'] == 'train'].time.max()}\n"
        + f"test_min_time == {vitrine[vitrine['segment'] == 'test'].time.min()}, "
        + f"test_max_time == {vitrine[vitrine['segment'] == 'test'].time.max()}\n"
        + f"unique_times_train == {vitrine[vitrine['segment'] == 'train'].time.unique().shape[0]}, \n"
        + f"unique_times_test == {vitrine[vitrine['segment'] == 'test'].time.unique().shape[0]}, \n"
    )

    return vitrine


# +
def split_dates(dates: List[int], count: int) -> Tuple[List[int], List[int]]:
    return dates[:count], dates[count:]

def sample_dates(dates_i: List[int], count: int) -> Tuple[List[int], List[int]]:
    dates = dates_i[::]
    np.random.shuffle(dates)
    return sorted(dates[:count]), sorted(dates[count:])

def get_dates_for_walk_forward(
    dates: List[int],
    train_val_ratio: float,
    val_size: int,
    num_folds: int,
) -> List[List[int]]:
    """
    :return: [[[train dates fold_1], [train-val dates fold_1], [val dates fold_1]], ...]
    """
    dates = sorted(dates)
    folds = []
    curr_train_dates = dates
    for i in range(num_folds):
        curr_train_dates, val_dates = split_dates(curr_train_dates, -val_size)
        train_val_size = int(train_val_ratio * len(curr_train_dates))
        train_dates, train_val_dates = sample_dates(curr_train_dates, -train_val_size)
        folds.append([train_dates, train_val_dates, val_dates])
    folds = folds[::-1]
    return folds

def limit_by_dates(
    vitrine: pd.DataFrame, nec_dates: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = vitrine[vitrine.time.isin(nec_dates)].reset_index(drop=True)
    df = df.sort_values(by=["time"]).reset_index(drop=True)
    return df


def walk_forward_splitting(
    base_vitrine: pd.DataFrame,
    train_val_size: int,
    val_size: int,
    num_folds: int,
    base: Path,
    folds_path: Path,
) -> None:
    folds = get_dates_for_walk_forward(
        sorted(list(set(base_vitrine.time))),
        train_val_size,
        val_size,
        num_folds,
    )

    (base / folds_path).mkdir()

    for num_folds in range(len(folds)):
        print(f"fold №{num_folds + 1}")
        fold = folds[num_folds]
        fold_types = ["train", "val-train", "val"]
        for ind, s in enumerate(fold):
            print(
                f"{fold_types[ind]}::\n len == {len(s)}, min == {min(s)}, max == {max(s)}"
            )

        # Limit by fold dates and save
        for ind, name in enumerate(["X", "train_val", "val"]):
            df = limit_by_dates(base_vitrine, fold[ind])
            df.to_parquet(
                base / folds_path / Path(f"{name}_{num_folds+1}.parquet.gzip"),
                compression="gzip",
            )

        print("\n\n")
