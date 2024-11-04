#!/bin/sh
set -e

MODEL=${1:-""}

docker compose --profile rag run --build rag_chat $MODEL