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
import urllib
import sys
import zipfile
import io

URLLIB=urllib
if sys.version_info >= (3, 0):
    URLLIB=urllib.request

class GLUEDownloader:
    def __init__(self, task, save_path):

        # Documentation - Download link obtained from here: https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py

        self.TASK2PATH = {"CoLA":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4',
                     "SST":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
                     "MRPC":{"mrpc_dev": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
                            "mrpc_train": 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt',
                            "mrpc_test": 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'},
                     "QQP":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5',
                     "STS":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5',
                     "MNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
                     "SNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df',
                     "QNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0',
                     "RTE":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb',
                     "WNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf',
                     "diagnostic":'https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D'}


        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.task = task

    def download(self):

        if self.task == 'MRPC':
            self.download_mrpc()
        elif self.task == 'diagnostic':
            self.download_diagnostic()
        else:
            self.download_and_extract(self.task)

    def download_and_extract(self, task):
        print("Downloading and extracting %s..." % task)
        data_file = "%s.zip" % task
        URLLIB.urlretrieve(self.TASK2PATH[task], data_file)
        print(data_file,"\n\n\n")
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(self.save_path)
        os.remove(data_file)
        print("\tCompleted!")

    def download_mrpc(self):
        print("Processing MRPC...")
        mrpc_dir = os.path.join(self.save_path, "MRPC")
        if not os.path.isdir(mrpc_dir):
            os.mkdir(mrpc_dir)

        mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
        mrpc_dev_file = os.path.join(mrpc_dir, "dev_ids.tsv")
        mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")

        URLLIB.urlretrieve(self.TASK2PATH["MRPC"]["mrpc_train"], mrpc_train_file)
        URLLIB.urlretrieve(self.TASK2PATH["MRPC"]["mrpc_test"], mrpc_test_file)
        URLLIB.urlretrieve(self.TASK2PATH["MRPC"]["mrpc_dev"], mrpc_dev_file)

        dev_ids = []
        with io.open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding='utf-8') as ids_fh:
            for row in ids_fh:
                dev_ids.append(row.strip().split('\t'))

        with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
                io.open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh, \
                io.open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
            header = data_fh.readline()
            train_fh.write(header)
            dev_fh.write(header)
            for row in data_fh:
                label, id1, id2, s1, s2 = row.strip().split('\t')
                if [id1, id2] in dev_ids:
                    dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
                else:
                    train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

        with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
                io.open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
            header = data_fh.readline()
            test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
            for idx, row in enumerate(data_fh):
                label, id1, id2, s1, s2 = row.strip().split('\t')
                test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
        print("\tCompleted!")