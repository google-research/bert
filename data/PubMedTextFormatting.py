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

import glob
import os
import pubmed_parser as pmp

class PubMedTextFormatting:
    def __init__(self, pubmed_path, output_filename, recursive = False):
        self.pubmed_path = pubmed_path
        self.recursive = recursive
        self.output_filename = output_filename


    # This puts one article per line
    def merge(self):
        print('PubMed path:', self.pubmed_path)

        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for filename in glob.glob(self.pubmed_path + '/*.xml', recursive=self.recursive):
                print('file:', filename)
                dicts_out = pmp.parse_medline_xml(filename)
                for dict_out in dicts_out:
                    if not dict_out['abstract']:
                        continue
                    try:
                        for line in dict_out['abstract'].splitlines():
                            if len(line) < 30:
                                continue
                            ofile.write(line.strip() + " ")
                        ofile.write("\n\n")
                    except:
                        ofile.write("\n\n")
                        continue
