#!/bin/bash
mkdir -p res
docker run -it --gpus=all -v ~/kcap-2023:/kcap-2023 gitlab.liu.se/rilca16/text2kg2023-rilca-util:latest bash