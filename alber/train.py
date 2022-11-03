# -*- coding: utf-8 -*-
# +
import pandas as pd
import h5py
import numpy as np
from typing import List, Tuple
from pathlib import Path
import lightgbm as lgb
import pickle
from sklearn.metrics import r2_score

from alber.load_data import read_order_book, read_trades, read_target
from alber.feature_generation import (
    book_preprocessor, 
    get_features_zscore, 
    get_features_ma, 
    get_features_stoch,
    retime_trades,
    decrease_mem_consuming
)
from alber.wf_splitting_data import create_oot, walk_forward_splitting, sample_dates, split_dates

# +
def get_only_time_features_vitrine(base: Path) -> pd.DataFrame:
    ob = read_order_book(base / Path('data.h5'))#.head(1_000)
    trades = read_trades(base / Path('data.h5'))#.head(1_000)
    print(f'ob.shape == {ob.shape}, trades.shape == {trades.shape}')
    
    ob = book_preprocessor(ob)
    
    # Preapre trade features
    trades['id'] = 0
    trades = get_features_zscore(trades)
    trades = get_features_ma(trades)
    trades = get_features_stoch(trades)
    trades = retime_trades(trades, ob)
    
    # Preprocess result features vitrine
    ob = ob.drop(['id'], axis=1)
    trades = trades.drop(['id'], axis=1)
    features = pd.merge(ob, trades, on=['time'])
    features = features.astype({'time': int})
    features = features.drop_duplicates(['time']).reset_index(drop=True)
    features = decrease_mem_consuming(features, ['time'])
    
    return features



def get_vitrine_for_train(base: Path) -> pd.DataFrame:
    features = get_only_time_features_vitrine(base)
    
    target = read_target(base / Path('result.h5'))
    features = pd.merge(features, target, on=['time'])
    
    return features


# +
def prepare_dataset_for_train(train_vitrine: pd.DataFrame, dates: List[int]) -> lgb.Dataset:
    df = train_vitrine.query(f'time == {dates}').reset_index(drop=True)
    
    ds = lgb.Dataset(
        df.drop(['time', 'target'], axis=1),
        df['target'],
        categorical_feature=[],
        free_raw_data=False,
    )
    
    return ds

def get_perfomance(model, test: lgb.Dataset) -> float:
    test = test.construct()
    data_test = test.get_data()
    predictions = model.predict(data_test)
    label_test = test.get_label()
    metric = r2_score(label_test, predictions)
    print(metric)
    
    return metric

def train(
    train_vitrine: pd.DataFrame, 
    name: str, 
    train_val_ratio: float,
    test_size: int,
    save_path: Path = Path('../saved_models'),
) -> float:
    curr_setting = {
        "verbose_eval": 50,
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
        "params": {
            "num_leaves": 131_072,
            "max_bin": 256,
            "learning_rate": 0.01,
            "objective": "regression",
            "metric": "rmse",
            "max_depth": 6,
            "feature_fraction": 1.0,
            "feature_fraction_bynode": 0.6,
            "bagging_fraction": 1.0
        },
    }
    
    # Splitting train_vitrine into train/val
    train_dates = sorted(list(set(train_vitrine.time)))
    train_dates, test_dates = split_dates(train_dates, -test_size)
    train_val_size = int(train_val_ratio * len(train_dates))
    train_dates, val_dates = sample_dates(train_dates, -train_val_size)
    print(f'len_train_dates == {len(train_dates)}, len_val_dates == {len(val_dates)}, len_test_dates == {len(test_dates)}')
    print(f'min_train_dates == {min(train_dates)}, min_val_dates == {min(val_dates)}, min_test_dates == {min(test_dates)}')
    print(f'max_train_dates == {max(train_dates)}, max_val_dates == {max(val_dates)}, max_test_dates == {max(test_dates)}')
    
    train = prepare_dataset_for_train(train_vitrine, train_dates)
    val = prepare_dataset_for_train(train_vitrine, val_dates)
    test = prepare_dataset_for_train(train_vitrine, test_dates)
    del train_vitrine
    
    # Train the model
    model = lgb.train(
        curr_setting["params"],
        train,
        valid_sets=[train, val],
        verbose_eval=curr_setting["verbose_eval"],
        num_boost_round=curr_setting["num_boost_round"],
        early_stopping_rounds=curr_setting["early_stopping_rounds"],
    )
    print(type(model))
    
    # Save the model
    save_path.mkdir(exist_ok=True)
    with open(save_path / Path(name + ".pkl"), "wb") as f:
        pickle.dump(model, f, protocol=2)
        
    
    return get_perfomance(model, test)
