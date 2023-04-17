import random
import numpy as np
import torch
import sys
import os


def set_folder(foldername):
    if '/notebook' in sys.path:
        folder_path = f'/notebook/personal/ksuchoi216/{foldername}'
        print('='*60)
        if folder_path not in sys.path:
            sys.path.insert(0, folder_path)
            os.chdir(folder_path)
            print(sys.path)
        print('='*60)


def set_seed(hash_str: str):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
