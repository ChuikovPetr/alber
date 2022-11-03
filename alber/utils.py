import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


def get_setting(adding_arguments: List[List[str]] = []):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", help="Activate debug regim", action="store_true"
    )
    parser.add_argument(
        "-in_name", help="Input directory", action="store", default="data"
    )
    parser.add_argument(
        "-out_name", help="Output directory", action="store", default="data"
    )

    for add in adding_arguments:
        parser.add_argument(add[0], help=add[1], default=add[2], action="store")

    args = parser.parse_args()

    curr_dir = Path(__file__).resolve().parent
    input_dir = Path(args.in_name)
    output_dir = Path(args.out_name)
    # if not output_dir.is_dir():
    #    output_dir.mkdir(parents=True)

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s :: %(filename)s :: %(funcName)s :: %(message)s",
        )

    settings = {
        "args": args,
        "curr_dir": curr_dir,
        "in_name": input_dir,
        "out_name": output_dir,
        "model_dir": curr_dir.parent / Path("saved_models"),
    }
    return settings
