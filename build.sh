#!/bin/bash
CONTAINER_NAME='entity-recontextualization'
DATE=$(date '+%F')
cd images/base
docker build -t $CONTAINER_NAME:latest .