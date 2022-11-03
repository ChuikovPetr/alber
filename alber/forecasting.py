# -*- coding: utf-8 -*-
# +
import pandas as pd
import h5py
from pathlib import Path
import pickle

from alber.load_data import read_order_book, read_trades
from alber.feature_generation import (
    book_preprocessor, 
    get_trend_features, 
    get_stoch,
    retime_trades,
    decrease_mem_consuming
)


# -

def get_nec_features_vitrine(data_path: Path) -> pd.DataFrame:
    ob = read_order_book(data_path)#.head(1_000)
    trades = read_trades(data_path)#.head(1_000)
    print(f'ob.shape == {ob.shape}, trades.shape == {trades.shape}')
    
    ob = book_preprocessor(ob)
    
    # Preapre trade features
    trades['id'] = 0
    trades = get_trend_features(
        trades,
        'price',
        ['id'],
        [1, 5, 10, 40, 80]
    )
    trades = get_trend_features(
        trades,
        'order_count',
        ['id'],
        [1, 80]
    )
    trades = get_stoch(trades, [21, 1, 3], 'price', ['id'])
    trades['Money'] = trades['price'] * trades['size']
    trades = retime_trades(trades, ob)
    
    # Preprocess result features vitrine
    ob = ob.drop(['id'], axis=1)
    trades = trades.drop(['id'], axis=1)
    features = pd.merge(ob, trades, on=['time'])
    features = features.astype({'time': int})
    features = features.drop_duplicates(['time']).reset_index(drop=True)
    features = decrease_mem_consuming(features, ['time'])
    
    return features


# +
def load_model(name: str, folder_path: Path):
    with open(folder_path / Path(name + ".pkl"), "rb") as f:
        return pickle.load(f)
    
    
def create_result_dataset(path: Path, score: pd.Series, ts: pd.Series) -> None:
    f = h5py.File(path,'w')
    grp = f.create_group("Return")
   
    Score = grp.create_dataset("Score", score.shape, dtype='float')
    TS = grp.create_dataset("TS", ts.shape, dtype='float')
   
    Score = score
    TS = ts
   
    f.close()
