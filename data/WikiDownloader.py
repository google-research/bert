# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import bz2
import os
import urllib.request
import sys
import subprocess

class WikiDownloader:
    def __init__(self, language, save_path):
        self.save_path = save_path + '/wikicorpus_' + language

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.language = language
        self.download_urls = {
            'en' : 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
            'zh' : 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
        }

        self.output_files = {
            'en' : 'wikicorpus_en.xml.bz2',
            'zh' : 'wikicorpus_zh.xml.bz2'
        }


    def download(self):
        if self.language in self.download_urls:
            url = self.download_urls[self.language]
            filename = self.output_files[self.language]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + filename):
                print('** Download file already exists, skipping download')
            else:
                response = urllib.request.urlopen(url)
                with open(self.save_path + '/' + filename, "wb") as handle:
                    handle.write(response.read())

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping:', self.output_files[self.language])
            subprocess.run('bzip2 -dk ' + self.save_path + '/' + filename, shell=True, check=True)

        else:
            assert False, 'WikiDownloader not implemented for this language yet.'

