import logging
from math import ceil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# from ccf.analytics import paired_test_greedy_fs
from ccf.analytics import paired_test_greedy_fs
from ccf.datasets import get_initial_setting
from sklearn.metrics import r2_score

# from ccf.models import ModelLGB
from ccf.models_cust import ModelLGB
from ccf.preprocess import get_sample_2d_lgb
from sl2r.utils import get_setting


def create_block_var_lgb(
    features_name: str,
    count_folds: int,
    experiment_name: str,
    current_feature_list: str,
    dataset: str,
    pred_iter_perf: str,
    input_dir: Path,
    max_bin: int = 255,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    min_data_in_leaf: int = 20,
) -> None:
    keys = ["time"]

    folds = []
    for i in range(count_folds):
        folds.append(
            [
                input_dir / Path(f"X_{i+1}.parquet.gzip"),
                input_dir / Path(f"train_val_{i+1}.parquet.gzip"),
                input_dir / Path(f"val_{i+1}.parquet.gzip"),
                input_dir.parent / Path(f"{features_name}.parquet.gzip"),
            ]
        )

    count_obs_train = None
    count_obs_val = None
    count_obs_val_train = None

    schema = pq.read_schema(
        input_dir.parent / Path(f"{features_name}.parquet.gzip"), memory_map=True
    )
    list_candidates = list(set(schema.names) - set(current_feature_list) - set(keys) - set(['__index_level_0__']))
    list_candidates.sort()

    selection_rule = {"field_name": "rel_diff_macro_lower_boot_95", "ascending": False}

    dict_fields, _, _ = get_initial_setting(
        input_dir.parent / Path(f"{features_name}.parquet.gzip"), count_cuttoff=0
    )

    analytics_path = input_dir / Path("analytics")

    try:
        pred_iter_perf = pred_iter_perf.replace("M", "-")
        pred_iter_perf = list(map(lambda x: float(x), pred_iter_perf.split(",")))
    except AttributeError:
        pred_iter_perf = None

    model_class = lambda train_matrix_shape, name: ModelLGB(
        save_path=input_dir.parent / Path("saved_models"),
        name=name,
        metric=r2_score,
        learning_setting={
            "verbose_eval": 5,
            "num_boost_round": 50,
            "early_stopping_rounds": 5,
            "params": {
                "num_leaves": 131_072,
                "max_bin": 256,
                "learning_rate": 0.01,
                "objective": "regression",
                "metric": "rmse",
                "max_depth": 6,
                "feature_fraction": 1.0,
                "feature_fraction_bynode": 1.0,
            },
        },
    )

    get_sample_func = lambda possible_feature_list, base_path, count_obs, scaler, features_path: get_sample_2d_lgb(
        possible_feature_list,
        base_path,
        count_obs,
        features_path,
        categoricals=[],
        experiment_name=dataset,
        keys=["time"],
    )

    paired_test_greedy_fs(
        current_feature_list,
        list_candidates,
        dict_fields,
        folds,
        count_obs_train,
        count_obs_val,
        experiment_name,
        model_class,
        analytics_path,
        selection_rule,
        get_sample_func,
        count_obs_val_train=count_obs_val_train,
        print_iteration=False,
        one_iter=True,
        count_iteration=5,
        pred_iter_perf=pred_iter_perf,
    )

    return None


def one_factor_analyse(
    count_folds: int,
    count_f_in_iter: int,
    features_name: str,
    current_feature_list: str,
    experiment_name: str,
    pred_iter_perf: str,
    input_dir: Path,
    output_dir: Path,
) -> None:
    if len(current_feature_list) == 0:
        current_feature_list = []
    else:
        current_feature_list = current_feature_list.split(",")

    features_path = input_dir.parent / Path(f"{features_name}.parquet.gzip")

    schema = pq.read_schema(features_path, memory_map=True)
    keys = ["time"]
    feature_names = list(set(schema.names) - set(current_feature_list) - set(keys) - set(['__index_level_0__']))
    feature_names.sort()

    count_iter = ceil(len(feature_names) / count_f_in_iter)
    for i in range(count_iter):
        cur_feature_names = feature_names[:count_f_in_iter] + current_feature_list
        feature_names = feature_names[count_f_in_iter:]
        logging.debug(
            f"{i+1}/{count_iter}:: Count cur_feature_names: {len(cur_feature_names)}; Count feature_names: {len(feature_names)}"
        )

        # Create features_buf.parquet.gzip(nrows=1) and dataset for count_f_in_iter features
        for fold_num in range(count_folds):
            for index, base_path in enumerate(
                [
                    input_dir / Path(f"X_{fold_num+1}.parquet.gzip"),
                    input_dir / Path(f"train_val_{fold_num+1}.parquet.gzip"),
                    input_dir / Path(f"val_{fold_num+1}.parquet.gzip"),
                ]
            ):
                if index == 2:
                    dataset = get_sample_2d_lgb(
                        list_features=cur_feature_names,
                        base_path=base_path,
                        count_obs=None,
                        features_path=features_path,
                        categoricals=[],
                        experiment_name=f"{experiment_name}_cur",
                        keys=["time"],
                    )
                elif index == 0:
                    dataset = get_sample_2d_lgb(
                        list_features=cur_feature_names,
                        base_path=base_path,
                        count_obs=None,
                        features_path=features_path,
                        categoricals=[],
                        experiment_name=f"{experiment_name}_cur",
                        keys=["time"],
                    )
                elif index == 1:
                    dataset = get_sample_2d_lgb(
                        list_features=cur_feature_names,
                        base_path=base_path,
                        count_obs=None,
                        features_path=features_path,
                        categoricals=[],
                        experiment_name=f"{experiment_name}_cur",
                        keys=["time"],
                    )


                del dataset
            logging.debug(f"{fold_num+1} folds:: dataset creating - complete")
        features = pd.read_parquet(features_path)
        features = features.head(1)[keys + cur_feature_names]
        # features.to_csv(input_dir.parent / Path(f"features_buf.parquet.gzip"), index=False)
        features.to_parquet(
            input_dir.parent / Path(f"features_buf.parquet.gzip"), compression="gzip"
        )
        del features

        create_block_var_lgb(
            "features_buf",
            count_folds,
            f"one_f_{i+1}",
            current_feature_list,
            f"{experiment_name}_cur",
            pred_iter_perf,
            input_dir,
        )
        one_f_cur = pd.read_csv(
            input_dir / Path("analytics") / Path(f"block_vars_one_f_{i+1}.csv")
        )

        try:
            res_one_f = pd.concat([res_one_f, one_f_cur])
        except:
            res_one_f = one_f_cur
        res_one_f = res_one_f.sort_values(
            by=[
                "count_boot_le_one",
                "rel_diff_macro_lower_boot_95",
                "macro_lower_boot_95",
            ],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        res_one_f.to_csv(
            input_dir
            / Path("analytics")
            / Path(f"block_vars_{experiment_name}.csv"),
            index=False,
        )

        # Delete excess
        for fold_num in range(count_folds):
            for segment in ["X", "train_val", "val"]:
                (
                    input_dir
                    / Path(f"{experiment_name}_cur_{segment}_{fold_num+1}.parquet._dataset.pkl")
                ).unlink()
        (
            input_dir / Path("analytics") / Path(f"block_vars_one_f_{i+1}.csv")
        ).unlink()

    return None

# python one_factor.py -count_folds 10 -count_f_in_iter 50 -in_name ../../Storage/sl2r/ret_1_10_folds_exp -out_name ../../Storage/sl2r/ret_1_10_folds_exp -d &> file
if __name__ == "__main__":
    settings = get_setting(
        [
            ["-count_folds", "Count of folds", ""],
            ["-count_f_in_iter", "Count considered feature in one iteration", ""],
            ["-feature_name", "Path of feature df", "features"],
            ["-cfl", "Current feature list", ""],
            ["-experiment_name", "Name of experiment", "one_f"],
            ["-pred_iter_perf", "Perfomance in previous iteration", None],
        ]
    )

    count_folds = int(eval(settings["args"].count_folds))
    count_f_in_iter = int(eval(settings["args"].count_f_in_iter))
    features_name = settings["args"].feature_name
    current_feature_list = settings["args"].cfl
    experiment_name = settings["args"].experiment_name
    pred_iter_perf = settings["args"].pred_iter_perf

    one_factor_analyse(
        count_folds,
        count_f_in_iter,
        features_name,
        current_feature_list,
        experiment_name,
        pred_iter_perf,
        settings["in_name"],
        settings["out_name"],
    )
