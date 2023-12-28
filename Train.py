# coding: utf-8

import argparse
import os
import yaml
import Run
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="Configs/Fast_dl_sim_div2k.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)
    os.makedirs(config['TRAIN']['OUTPUT_DIR'], exist_ok=True)
    shutil.copy2(args.config_path, config['TRAIN']['OUTPUT_DIR'])
    Run.main(config)
