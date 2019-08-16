#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:24:39 2019

@author: zjy
"""
import random
import requests
import argparse
parser = argparse.ArgumentParser(description='Tensorflow serving client test')
parser.add_argument('-url', type=str, default='http://0.0.0.0:8901/v1/models/my_model:predict',
                    help='url of server that supports RESTFUL')
args = parser.parse_args()


if __name__ == '__main__':
    # 1. check server state by GET method
    result = requests.get(args.url[:-len(":predict")])
    result_json = result.json()
    print(result_json)
    assert result_json["model_version_status"][0]["state"] == "AVAILABLE"
    """  Normally, this should give you the following result
    {
        "model_version_status":
        [
            {
               "version": "1",
               "state": "AVAILABLE",
               "status": {
                    "error_code": "OK",
                    "error_message": ""
                }
            }
        ]
    }
    """

    # 2. Send data by POST method
    """ review of model input/output information (showed by saved_model_cli in run_server.sh)
    The given SavedModel SignatureDef contains the following input(s):
        inputs['x'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1)
            name: x:0
        inputs['y'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1)
            name: y:0
    The given SavedModel SignatureDef contains the following output(s):
        outputs['sum'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1)
            name: add:0
    """
    input_ids = [101, 2572, 3217, 5831, 5496, 2010, 2567, 1010, 3183, 2002, 2170, 1000, 1996, 7409, 1000, 1010, 1997, 9969, 4487, 23809, 3436, 2010, 3350, 1012, 102, 7727, 2000, 2032, 2004, 2069, 1000, 1996, 7409, 1000, 1010, 2572, 3217, 5831, 5496, 2010, 2567, 1997, 9969, 4487, 23809, 3436, 2010, 3350, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label_ids = 1
    sample_data = []
    for i in range(1):  # sample 4 (x, y) pairs
        single_sample = {"input_ids": input_ids, "input_mask": input_mask, 'segment_ids': segment_ids, 'label_ids': label_ids}
        sample_data.append(single_sample)
    # the simple data format is {"instances": a list of single_sample}, where
    # all instances will be batched at server side
    inputs = {"instances": sample_data}
    result = requests.post(args.url, json=inputs)

    print("inputs:")
    print(inputs)
    print()
    print("outputs:")
    print(result.json())
