#!/bin/sh

MODEL=${1:-""}

docker compose up -d
docker compose --profile basic_chat up --build -d
docker compose --profile basic_chat exec basic_chat python chat.py $MODEL