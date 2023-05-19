import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))

ARGS_FILEPATHS = [os.path.join(ROOT_DIR,f) for f in [
    'training_config.json',
    'segmentation_config.json',
]]