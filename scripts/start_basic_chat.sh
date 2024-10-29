#!/bin/sh

MODEL=${1:-""}

docker compose --profile basic up --build -d
docker compose --profile basic exec basic_chat python chat.py $MODEL