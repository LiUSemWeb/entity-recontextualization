#!/bin/bash
PATH=/usr/bin:$PATH
DOCKER_HOST=unix:///run/user/10013/docker.sock
CONTAINER=entity-recontextualization
docker kill --signal=SIGINT semantics
docker stop semantics && docker rm semantics
# task_name = sys.argv[1] if len(sys.argv) > 1 else "docred"
# data_set = sys.argv[2] if len(sys.argv) > 2 else "dev"
# model = sys.argv[3] if len(sys.argv) > 3 else "bert-large-cased"
# num_passes = int(sys.argv[4]) if len(sys.argv) > 4 else 3
# resdir = sys.argv[5] if len(sys.argv) > 5 else "res"
# max_batch = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
# start_at = int(sys.argv[7]) if len(sys.argv) > 7 else 0
# run_exp.sh biored train biobert 2 1000 0 && docker logs --follow semantics
# python EntitySubstituteTest.py biored train biobert 2 res 1000 0
docker run -d -i -v $pwd:/entity-recontextualization --gpus=all --name=semantics $CONTAINER:latest bash -c "cd /entity-recontextualization && python EntitySubstituteTest.py $1 $2 $3 $4 res $5 $6"