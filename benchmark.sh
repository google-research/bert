#!/usr/bin/env bash
./wrk  -t 4 -c 128 -d 20s --timeout=10s -s /data/nfsdata/home/wuwei/study/bert/benchmark.lua http://0.0.0.0:8901/v1/models/my_model:predict