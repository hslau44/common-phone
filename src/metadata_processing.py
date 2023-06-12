import os
import pandas as pd
from praatio import textgrid


root = 'CP'
metadata_filenames = ['meta','train','dev','test',]


def get_lang_metadata(root,lang):
    md = []
    for fname in metadata_filenames:
        fp = os.path.join(root,lang,fname+'.csv')
        md += [pd.read_csv(fp)]
    metadata = pd.concat(md[1:])
    metadata = pd.merge(metadata,md[0])
    metadata = metadata.drop('id',axis=1)
    return metadata


def get_metadata(root):
    langs = os.listdir(root)
    return pd.concat([get_lang_metadata(root,lang=l) for l in langs])


def get_phonetic_info(root,lang,fname,tiername='MAU'):
    fp = os.path.join(root,lang,'grids',fname+'.TextGrid')
    tierdict = textgrid.openTextgrid(fp,False).tierDict
    if tiername is not None:
        tier = tierdict[tiername]
        interval_df = pd.DataFrame(tier.entryList)
        phonetic_detail = {col:interval_df[col].tolist() for col in interval_df.columns}   
    else:
        phonetic_detail = {}
        for tiername,tier in tierdict.items():
            interval_df = pd.DataFrame(tier.entryList)
            interval_dict = {col:interval_df[col].tolist() for col in interval_df.columns}
            phonetic_detail[tiername] = interval_dict 
    return phonetic_detail


def applyfunc_src_fp(row,src_root,fmt='wav'):
    fname = row['audio_file'].split('.')[0] +'.' + fmt
    lang = row['locale']
    fp = os.path.join(src_root,lang,fmt,fname)
    return fp


def applyfunc_target_fp(row,target_root,fmt='wav',add_lang=False):
    fname = row['audio_file'].split('.')[0] +'.' + fmt
    split = row['set']
    lang = row['locale']
    if add_lang:
        fp = os.path.join(target_root,split,lang,fname)
    else:
        fp = os.path.join(target_root,split,fname)
    return fp


def applyfunc_phone(row,root,tiername='MAU'):
    fname = row['audio_file'].split('.')[0]
    lang = row['locale']
    phonetic_detail = get_phonetic_info(root,lang,fname,tiername)
    return phonetic_detail


def reorder_columns(metadata):
    target_fnames = metadata['file_name']
    metadata = metadata.drop(columns=['file_name'])
    metadata.insert(loc=0, column='file_name', value=target_fnames)
    return metadata
    


def process_metadata(metadata):
    ### rename audio file column ###
    metadata = metadata.rename({'audio file':'audio_file'},axis=1)
    ### remove .mp3 ###
    target_root = 'data'
    metadata['phonetic_detail'] = metadata.apply(lambda row: applyfunc_phone(row,root),axis=1)
    metadata['src_file_name'] = metadata.apply(lambda row: applyfunc_src_fp(row,root,fmt='wav'),axis=1)
    metadata['file_name'] = metadata.apply(lambda row: applyfunc_target_fp(row,target_root,fmt='wav'),axis=1)
    ### add phonetic_detail from textgrid ###
    metadata = reorder_columns(metadata)
    metadata = metadata.drop(columns='audio_file',axis=1)
    return metadata


if __name__ == "__main__":
    metadata = get_metadata(root)
    metadata = process_metadata(metadata)
    metadata.to_json('metadata.json',orient='table',index=False)
    #metadata.to_csv('metadata.csv')



