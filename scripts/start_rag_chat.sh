#!/bin/sh

MODEL=${1:-""}

docker compose --profile rag up --build -d
docker compose --profile rag exec rag_chat python chat.py $MODEL