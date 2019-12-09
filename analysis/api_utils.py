import os
import sys
import logging
import configparser
from pprint import pprint
import argparse

import numpy as np


def setup_logging(in_dir):
    fname = os.path.join(in_dir, 'nc_output.log')
    if os.path.isfile(fname):
        open(fname, 'w').close()
    logging.basicConfig(
        filename=fname, level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)


def print_config(config, msg=""):
    if msg is not "":
        print(msg)
    """Prints the contents of a config file"""
    config_dict = [{x: tuple(config.items(x))} for x in config.sections()]
    pprint(config_dict, width=120)


def read_cfg(location, verbose=True):
    config = configparser.ConfigParser()
    config.read(location)

    if verbose:
        print_config(config, "Program started with configuration")
    return config


def parse_args(verbose=True):
    parser = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    args, unparsed = parser.parse_known_args()

    if len(unparsed) is not 0:
        print("Unrecognised command line argument passed")
        print(unparsed)
        exit(-1)

    if verbose:
        if len(sys.argv) > 1:
            print("Command line arguments", args)
    return args


def make_path_if_not_exists(fname):
    """Makes directory structure for given fname"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)


def make_dir_if_not_exists(dirname):
    """Makes directory structure for given fname"""
    os.makedirs(dirname, exist_ok=True)


def save_mixed_dict_to_csv(in_dict, out_dir, out_name="results.csv"):
    """
    Save a dictionary with mixed value types to a csv.

    Currently dict, np.ndarray, and list are supported values.

    Args:
        in_dict (dict): The dictionary to save to a csv.
        out_dir (str): The directory to save the csv to.
        out_name (str, optional): Defaults to "results.csv".

    Returns:
        None
    """
    def arr_to_str(name, arr):
        out_str = name
        for val in arr:
            if isinstance(val, str):
                out_str = "{},{}".format(out_str, val)
            else:
                out_str = "{},{:2f}".format(out_str, val)
        return out_str

    out_loc = os.path.join(out_dir, out_name)
    make_dir_if_not_exists(out_loc)
    with open(out_loc, "w") as f:
        for key, val in in_dict.items():
            if isinstance(val, dict):
                out_str = arr_to_str(key, val.values())
            elif isinstance(val, np.ndarray):
                out_str = arr_to_str(key, val.flatten())
            elif isinstance(val, list):
                out_str = arr_to_str(key, val)
            else:
                print("Unrecognised type {} quitting".format(
                    type(val)
                ))
                exit(-1)
            f.write(out_str + "\n")
