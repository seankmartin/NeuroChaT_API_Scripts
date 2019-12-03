import os
import sys
import logging
import configparser
from pprint import pprint
import argparse


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
