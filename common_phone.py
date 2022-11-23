# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""
TODO: Add a description here.

"""


import csv
import json
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

_DATA_DIR = "TODO"

_METADATA_DIR = "TODO"

_DATA_URLS = {
    "train": f"{_DATA_DIR}/train.tar.gz",
    "validation": f"{_DATA_DIR}/validation.tar.gz",
    "test": f"{_DATA_DIR}/test.tar.gz",
}

_METADATA_URLS = {
    "train": f"{_METADATA_DIR}/train.json",
    "validation": f"{_METADATA_DIR}/validation.json",
    "test": f"{_METADATA_DIR}/test.json",
}

_URLS = {
    "data":_DATA_URLS,
    "metadata":_METADATA_URLS
}


class CommonPhoneConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonPhone."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(CommonPhoneConfig, self).__init__(**kwargs)
    


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class CommonPhone(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        CommonPhoneConfig(name="_all_", version=VERSION, description="All"),
        CommonPhoneConfig(name="de", version=VERSION, description="German"),
        CommonPhoneConfig(name="en", version=VERSION, description="English"),
        CommonPhoneConfig(name="es", version=VERSION, description="Spanish"),
        CommonPhoneConfig(name="fr", version=VERSION, description="French"),
        CommonPhoneConfig(name="it", version=VERSION, description="Italian"),
        CommonPhoneConfig(name="ru", version=VERSION, description="Russian"),
    ]

    DEFAULT_CONFIG_NAME = "_all_" 
    
    
    @property
    def manual_download_instructions(self):
        return (
            "To use CommonPhone you have to download it manually. "
            "Please create an account and download the dataset from https://catalog.ldc.upenn.edu/LDC93S1 \n"
            "Then extract all files in one folder and load the dataset with: "
            "`datasets.load_dataset('timit_asr', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        
        features = datasets.Features(
            {
                'file_name': datasets.Value("string"), 
                'text': datasets.Value("string"), 
                'gender': datasets.Value("string"), 
                'age': datasets.Value("string"), 
                'locale': datasets.Value("string"), 
                'accent': datasets.Value("string"), 
                'set': datasets.Value("string"),
                "phonetic_detail": datasets.Sequence(
                    {
                        "start": datasets.Value("int64"),
                        "end": datasets.Value("int64"),
                        "label": datasets.Value("string"),
                    }
                ),
                'src_file_name': datasets.Value("string"), 
                'audio':datasets.Audio(sampling_rate=16000),
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features, 
            supervised_keys=("file_name", "text"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        dl_manager.download_config.ignore_url_params = True
        audio_path = {}
        local_extracted_archive = {}
        metadata_path = {}
        
        split_type = {"train": datasets.Split.TRAIN, "validation": datasets.Split.TEST, "test": datasets.Split.TEST}
        
        for split,url in _DATA_URLS.items():
            audio_path[split] = dl_manager.download(url)
            local_extracted_archive[split] = dl_manager.extract(audio_path[split]) if not dl_manager.is_streaming else None
        
        
        for split,url in _METADATA_URLS.items():
            metadata_path[split] = dl_manager.download(url)
        
        path_to_clips = "CP"
        
        return [
            datasets.SplitGenerator(
                name=split_type[split],
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive[split], # 
                    "audio_files": dl_manager.iter_archive(audio_path[split]), # all audio path in str 
                    "metadata_path": dl_manager.download_and_extract(metadata_path[split]),
                    "path_to_clips": path_to_clips,
                },
            ) for split in split_type
        ]
    

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self, 
        local_extracted_archive,
        audio_files,
        metadata_path,
        path_to_clips,
    ):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        metadata = {}
        with open(metadata_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.config.name == "_all_" or self.config.name == row["language"]:
                    
                    phonemes = read_dic_from_str(row["phonetic_detail"])
                
                    example = {
                        'file_name': row['file_name'], 
                        'text': row['text'], 
                        'gender': row['gender'], 
                        'age': row['age'], 
                        'locale': row['locale'], 
                        'accent': row['accent'], 
                        'set': row['set'],
                        "phonetic_detail": phonemes
                        'src_file_name': row['src_file_name'], 
                    }
                    
                    metadata[row['file_name']] = example
        
        id_ = 0
        for path, f in audio_files:
            if path in metadata:
                example = metadata[path]
                # set the audio feature and the path to the extracted file
                path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                example["audio"] = {"path": path, "bytes": f.read()}
                example["path"] = path
                yield id_, example
                id_ += 1

                
def _read_dic_from_str(string):
    return eval(string)