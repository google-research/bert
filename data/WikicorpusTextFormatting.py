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

class WikicorpusTextFormatting:
    def __init__(self, wiki_path, output_filename, recursive = False):
        self.wiki_path = wiki_path
        self.recursive = recursive
        self.output_filename = output_filename


    # This puts one article per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for dirname in glob.glob(self.wiki_path + '/*/', recursive=False):
                for filename in glob.glob(dirname + 'wiki_*', recursive=self.recursive):
                    print(filename)
                    article_lines = []
                    article_open = False

                    with open(filename, mode='r', newline='\n') as file:
                        for line in file:
                            if '<doc id=' in line:
                                article_open = True
                            elif '</doc>' in line:
                                article_open = False
                                for oline in article_lines[1:]:
                                    if oline != '\n':
                                        ofile.write(oline.rstrip() + " ")
                                ofile.write("\n\n")
                                article_lines = []
                            else:
                                if article_open:
                                    article_lines.append(line)