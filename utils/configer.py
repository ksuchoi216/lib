import yaml
import os
import json
import pprint


class Configer:
    def __init__(self, config_name, config_dir='./configs'):
        self.config_path = f'{config_dir}/{config_name}.json'
        print(f'config_path: {self.config_path}')

    def get(self):
        isconfig = os.path.exists(self.config_path)
        if isconfig:
            with open(self.config_path, "r") as path:
                cfg = json.load(path)

            general = {}
            for k, v in cfg.items():
                # print(f'{k:10}: {v}')
                if not isinstance(v, dict):
                    general[k] = v

            for k, v in cfg.items():
                if k == 'hps':
                    continue

                if isinstance(v, dict):
                    cfg[k]['general'] = general
            self.cfg = cfg
        else:
            return print("There is NOT a config file")

        return self.cfg


def load_json(config_name):
    config_dir = f'./configs/{config_name}.json'
    print(f'config_dir: {config_dir}')
    try:
        with open(config_dir, "r") as path:
            cfg = json.load(path)
        pprint.pprint(cfg)
        return cfg
    except FileNotFoundError:
        print('File does NOT exist')


def load_yaml():
    with open('./configs/baseline.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)
