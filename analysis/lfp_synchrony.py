import os

from api_utils import read_cfg, parse_args


def main(cfg, args, **kwargs):
    pass


if __name__ == "__main__":
    """Parse args and cfg and send to main."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "lfp_synchrony.cfg")
    config = read_cfg(config_path)
    args = parse_args()
    main(config, args)
