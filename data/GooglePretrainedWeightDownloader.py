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

import hashlib
import os
import urllib.request
import zipfile

class GooglePretrainedWeightDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/google_pretrained_weights'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Download urls
        self.model_urls = {
            'bert_base_uncased': ('https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip', 'uncased_L-12_H-768_A-12.zip'),
            'bert_large_uncased': ('https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip', 'uncased_L-24_H-1024_A-16.zip'),
            'bert_base_cased': ('https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip', 'cased_L-12_H-768_A-12.zip'),
            'bert_large_cased': ('https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip', 'cased_L-24_H-1024_A-16.zip'),
            'bert_base_multilingual_cased': ('https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip', 'multi_cased_L-12_H-768_A-12.zip'),
            'bert_large_multilingual_uncased': ('https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip', 'multilingual_L-12_H-768_A-12.zip'),
            'bert_base_chinese': ('https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip', 'chinese_L-12_H-768_A-12.zip')
        }

        # SHA256sum verification for file download integrity (and checking for changes from the download source over time)
        self.bert_base_uncased_sha = {
            'bert_config.json': '7b4e5f53efbd058c67cda0aacfafb340113ea1b5797d9ce6ee411704ba21fcbc',
            'bert_model.ckpt.data-00000-of-00001': '58580dc5e0bf0ae0d2efd51d0e8272b2f808857f0a43a88aaf7549da6d7a8a84',
            'bert_model.ckpt.index': '04c1323086e2f1c5b7c0759d8d3e484afbb0ab45f51793daab9f647113a0117b',
            'bert_model.ckpt.meta': 'dd5682170a10c3ea0280c2e9b9a45fee894eb62da649bbdea37b38b0ded5f60e',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        }

        self.bert_large_uncased_sha = {
            'bert_config.json': 'bfa42236d269e2aeb3a6d30412a33d15dbe8ea597e2b01dc9518c63cc6efafcb',
            'bert_model.ckpt.data-00000-of-00001': 'bc6b3363e3be458c99ecf64b7f472d2b7c67534fd8f564c0556a678f90f4eea1',
            'bert_model.ckpt.index': '68b52f2205ffc64dc627d1120cf399c1ef1cbc35ea5021d1afc889ffe2ce2093',
            'bert_model.ckpt.meta': '6fcce8ff7628f229a885a593625e3d5ff9687542d5ef128d9beb1b0c05edc4a1',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        }

        self.bert_base_cased_sha = {
            'bert_config.json': 'f11dfb757bea16339a33e1bf327b0aade6e57fd9c29dc6b84f7ddb20682f48bc',
            'bert_model.ckpt.data-00000-of-00001': '734d5a1b68bf98d4e9cb6b6692725d00842a1937af73902e51776905d8f760ea',
            'bert_model.ckpt.index': '517d6ef5c41fc2ca1f595276d6fccf5521810d57f5a74e32616151557790f7b1',
            'bert_model.ckpt.meta': '5f8a9771ff25dadd61582abb4e3a748215a10a6b55947cbb66d0f0ba1694be98',
            'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
        }

        self.bert_large_cased_sha = {
            'bert_config.json': '7adb2125c8225da495656c982fd1c5f64ba8f20ad020838571a3f8a954c2df57',
            'bert_model.ckpt.data-00000-of-00001': '6ff33640f40d472f7a16af0c17b1179ca9dcc0373155fb05335b6a4dd1657ef0',
            'bert_model.ckpt.index': 'ef42a53f577fbe07381f4161b13c7cab4f4fc3b167cec6a9ae382c53d18049cf',
            'bert_model.ckpt.meta': 'd2ddff3ed33b80091eac95171e94149736ea74eb645e575d942ec4a5e01a40a1',
            'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
        }

        self.bert_base_multilingual_cased_sha = {
            'bert_config.json': 'e76c3964bc14a8bb37a5530cdc802699d2f4a6fddfab0611e153aa2528f234f0',
            'bert_model.ckpt.data-00000-of-00001': '55b8a2df41f69c60c5180e50a7c31b7cdf6238909390c4ddf05fbc0d37aa1ac5',
            'bert_model.ckpt.index': '7d8509c2a62b4e300feb55f8e5f1eef41638f4998dd4d887736f42d4f6a34b37',
            'bert_model.ckpt.meta': '95e5f1997e8831f1c31e5cf530f1a2e99f121e9cd20887f2dce6fe9e3343e3fa',
            'vocab.txt': 'fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c',
        }

        self.bert_large_multilingual_uncased_sha = {
            'bert_config.json': '49063bb061390211d2fdd108cada1ed86faa5f90b80c8f6fdddf406afa4c4624',
            'bert_model.ckpt.data-00000-of-00001': '3cd83912ebeb0efe2abf35c9f1d5a515d8e80295e61c49b75c8853f756658429',
            'bert_model.ckpt.index': '87c372c1a3b1dc7effaaa9103c80a81b3cbab04c7933ced224eec3b8ad2cc8e7',
            'bert_model.ckpt.meta': '27f504f34f02acaa6b0f60d65195ec3e3f9505ac14601c6a32b421d0c8413a29',
            'vocab.txt': '87b44292b452f6c05afa49b2e488e7eedf79ea4f4c39db6f2f4b37764228ef3f',
        }

        self.bert_base_chinese_sha = {
            'bert_config.json': '7aaad0335058e2640bcb2c2e9a932b1cd9da200c46ea7b8957d54431f201c015',
            'bert_model.ckpt.data-00000-of-00001': '756699356b78ad0ef1ca9ba6528297bcb3dd1aef5feadd31f4775d7c7fc989ba',
            'bert_model.ckpt.index': '46315546e05ce62327b3e2cd1bed22836adcb2ff29735ec87721396edb21b82e',
            'bert_model.ckpt.meta': 'c0f8d51e1ab986604bc2b25d6ec0af7fd21ff94cf67081996ec3f3bf5d823047',
            'vocab.txt': '45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c',
        }

        # Relate SHA to urls for loop below
        self.model_sha = {
            'bert_base_uncased': self.bert_base_uncased_sha,
            'bert_large_uncased': self.bert_large_uncased_sha,
            'bert_base_cased': self.bert_base_cased_sha,
            'bert_large_cased': self.bert_large_cased_sha,
            'bert_base_multilingual_cased': self.bert_base_multilingual_cased_sha,
            'bert_large_multilingual_uncased': self.bert_large_multilingual_uncased_sha,
            'bert_base_chinese': self.bert_base_chinese_sha
        }

    # Helper to get sha256sum of a file
    def sha256sum(self, filename):
      h  = hashlib.sha256()
      b  = bytearray(128*1024)
      mv = memoryview(b)
      with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
          h.update(mv[:n])

      return h.hexdigest()

    def download(self):
        # Iterate over urls: download, unzip, verify sha256sum
        found_mismatch_sha = False
        for model in self.model_urls:
          url = self.model_urls[model][0]
          file = self.save_path + '/' + self.model_urls[model][1]

          print('Downloading', url)
          response = urllib.request.urlopen(url)
          with open(file, 'wb') as handle:
            handle.write(response.read())

          print('Unzipping', file)
          zip = zipfile.ZipFile(file, 'r')
          zip.extractall(self.save_path)
          zip.close()

          sha_dict = self.model_sha[model]
          for extracted_file in sha_dict:
            sha = sha_dict[extracted_file]
            if sha != self.sha256sum(file[:-4] + '/' + extracted_file):
              found_mismatch_sha = True
              print('SHA256sum does not match on file:', extracted_file, 'from download url:', url)
            else:
              print(file[:-4] + '/' + extracted_file, '\t', 'verified')

        if not found_mismatch_sha:
          print("All downloads pass sha256sum verification.")

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def listAvailableWeights(self):
        print("Available Weight Datasets")
        for item in self.model_urls:
            print(item)

    def listLocallyStoredWeights(self):
        pass

