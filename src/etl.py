import os
import argparse
import pandas as pd
from tqdm import tqdm
import shutil
from utils import write_json, read_json


def move_audio_file(metadata):
    err = []
    for index, row in tqdm(metadata.iterrows()):
        fp = row['src_file_name']
        dest = os.path.join(*row['file_name'].split(os.path.sep)[:-1])
        try:
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(fp,dest)
        except:
            err.append(fp)
    return err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    
    metadata = pd.read_json(args.file,orient='table')
    print(f'Found {len(metadata)} files, process start')
    err = move_audio_file(metadata)
    if len(err) > 0:
        err = pd.DataFrame(err)
        err.to_json('metadata_err.json',orient='table',index=False)
        print(f"Number of failed operation: {len(err)}, saved in 'metadata_err.json'")
    else:
        print('All files are transferred')