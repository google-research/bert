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
import glob
import gzip
import os
import urllib.request
import shutil
import sys

class PubMedDownloader:
    def __init__(self, subset, save_path):
        self.subset = subset
        # Modifying self.save_path in two steps to handle creation of subdirectories
        self.save_path = save_path + '/pubmed' + '/'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.save_path = self.save_path + '/' + subset

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.download_urls = {
            'baseline' : 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/',
            'daily_update' : 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/',
            'fulltext' : 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/',
            'open_access' : 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/'
        }


    def download(self):
        print('subset:', self.subset)
        url = self.download_urls[self.subset]
        self.download_files(url)
        self.extract_files()


    def download_files(self, url):
        url = self.download_urls[self.subset]
        output = os.popen('curl ' + url).read()

        if self.subset == 'fulltext' or self.subset == 'open_access':
            line_split = 'comm_use' if self.subset == 'fulltext' else 'non_comm_use'
            for line in output.splitlines():
                if line[-10:] == 'xml.tar.gz' and \
                        line.split(' ')[-1].split('.')[0] == line_split:
                    file = os.path.join(self.save_path, line.split(' ')[-1])
                    if not os.path.isfile(file):
                        print('Downloading', file)
                        response = urllib.request.urlopen(url + line.split(' ')[-1])
                        with open(file, "wb") as handle:
                            handle.write(response.read())

        elif self.subset == 'baseline' or self.subset == 'daily_update':
            for line in output.splitlines():
                if line[-3:] == '.gz':
                    file = os.path.join(self.save_path, line.split(' ')[-1])
                    if not os.path.isfile(file):
                        print('Downloading', file)
                        response = urllib.request.urlopen(url + line.split(' ')[-1])
                        with open(file, "wb") as handle:
                            handle.write(response.read())
        else:
            assert False, 'Invalid PubMed dataset/subset specified.'

    def extract_files(self):
        files = glob.glob(self.save_path + '/*.xml.gz')

        for file in files:
            print('file:', file)
            input = gzip.GzipFile(file, mode='rb')
            s = input.read()
            input.close()

            out = open(file[:-3], mode='wb')
            out.write(s)
            out.close()



