#!/bin/sh
set -e

MODEL=${1:-""}

docker compose --profile rag up --remove-orphans -d
docker compose --profile rag run --remove-orphans --build rag_chat chat.py $MODEL