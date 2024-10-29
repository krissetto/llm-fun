#!/bin/sh

docker compose --profile rag up --build -d
docker compose --profile rag exec rag_chat python main.py