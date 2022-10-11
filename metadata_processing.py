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


def applyfunc_fname(audio_file,root,fmt='wav'):
    fname = audio_file.split('.')[0]
#     lang = fname.split('_')[2]
#     fp = os.path.join(root,lang,fmt,fname+'.'+fmt)
    return fname


def applyfunc_phone(fname,root,tiername='MAU'):
#     fname = audio_file.split('.')[0]
    lang = fname.split('_')[2]
    phonetic_detail = get_phonetic_info(root,lang,fname,tiername)
    return phonetic_detail


if __name__ == "__main__":
    ### load and concatnate all file from available language as one metadata ###
    metadata = get_metadata(root)

    ### rename audio file column ###
    metadata = metadata.rename({'audio file':'audio_file'},axis=1)
    ### remove .mp3 ###
    metadata['audio_file'] = metadata['audio_file'].apply(lambda af: applyfunc_fname(af,root))
    ### add phonetic_detail from textgrid ###
    metadata['phonetic_detail'] = metadata['audio_file'].apply(lambda af: applyfunc_phone(af,root))

    ### write json file ###
    metadata.to_json('metadata.json',orient='table',index=False)
    # metadata.to_csv('metadata.csv')

    ### add list json for etl ###
    write_json(metadata['audio_file'].tolist(),'audio_files.json')
    # metadata = pd.read_json('metadata.json',orient='table')
    # print(metadata.shape)
    # metadata.head()

