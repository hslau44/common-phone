import os
import argparse
import json
from tqdm import tqdm
import shutil
from utils import write_json, read_json


def move_audio_file(root,fmt,ls_fname,dest):
    assert os.path.exists(dest), 'dest does not exit'
    err = []
    for fname in tqdm(ls_fname):
        lang = fname.split('_')[2]
        fp = os.path.join(root,lang,fmt,fname+'.'+fmt)
        try:
            shutil.copy(fp,dest)
        except:
            err.append(fp)
    write_json(err,'failed_files.json')
    output = f'Number of failed operation: {len(err)}'
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('fmt')
    parser.add_argument('file')
    parser.add_argument('dest')
    args = parser.parse_args()
    
    ls_fname = read_json(args.file)
    print(args.root,args.fmt,len(ls_fname),args.dest)
    move_audio_file(args.root,args.fmt,ls_fname,args.dest)