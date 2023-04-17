import os
import json
import pprint


class Config:
    def __init__(self, config_name):
        config_dir = f'.lib/clib/configs/config_{config_name}.json'
        isconfig = os.path.exists(config_dir)

        if isconfig:
            with open(config_dir, "r") as path:
                cfg = json.load(path)

            general = {}
            for k, v in cfg.items():
                # print(f'{k:10}: {v}')
                if not isinstance(v, dict):
                    general[k] = v

            for k, v in cfg.items():
                if k == 'hyperparameters':
                    continue

                if isinstance(v, dict):
                    cfg[k]['general'] = general
                print(f'{k:20}: {v}')

            print('hyperparameter: ')
            pprint.pprint(cfg['hyperparameters'])
            self.cfg = cfg

    def __call__(self):
        return self.cfg

    def get(self):
        return self.cfg


def load_json(config_name):
    config_dir = f'./configs/config_{config_name}.json'
    print(f'config_dir: {config_dir}')
    try:
        with open(config_dir, "r") as path:
            cfg = json.load(path)
        pprint.pprint(cfg)
        return cfg
    except FileNotFoundError:
        print('File does NOT exist')
