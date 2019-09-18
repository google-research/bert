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

import subprocess

class BooksDownloader:
    def __init__(self, save_path):
        self.save_path = save_path
        pass


    def download(self):
        bookscorpus_download_command = 'python3 /workspace/bookcorpus/download_files.py --list /workspace/bookcorpus/url_list.jsonl --out'
        bookscorpus_download_command += ' ' + self.save_path + '/bookscorpus'
        bookscorpus_download_command += ' --trash-bad-count'
        bookscorpus_download_process = subprocess.run(bookscorpus_download_command, shell=True, check=True)