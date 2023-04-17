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
