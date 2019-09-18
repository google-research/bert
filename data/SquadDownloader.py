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

class SquadDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/squad'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(self.save_path + '/v1.1'):
            os.makedirs(self.save_path + '/v1.1')

        if not os.path.exists(self.save_path + '/v2.0'):
            os.makedirs(self.save_path + '/v2.0')

        self.download_urls = {
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json' : 'v1.1/train-v1.1.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' : 'v1.1/dev-v1.1.json',
            'https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/' : 'v1.1/evaluate-v1.1.py',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json' : 'v2.0/train-v2.0.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json' : 'v2.0/dev-v2.0.json',
            'https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/' : 'v2.0/evaluate-v2.0.py',
        }

    def download(self):
        for item in self.download_urls:
            url = item
            file = self.download_urls[item]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + file):
                print('** Download file already exists, skipping download')
            else:
                response = urllib.request.urlopen(url)
                with open(self.save_path + '/' + file, "wb") as handle:
                    handle.write(response.read())


