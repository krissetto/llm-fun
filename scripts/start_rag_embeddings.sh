#!/bin/sh
set -e

MODEL=${1:-""}

docker compose --profile rag run --remove-orphans --build rag_chat create_embeddings.py $MODEL