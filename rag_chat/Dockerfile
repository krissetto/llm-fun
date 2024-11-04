# syntax=docker/dockerfile:1

FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,id=rag_chat \
    pip install -r requirements.txt


# Stage 2: Runtime Environment
FROM python:3.12-slim
WORKDIR /app
COPY --link --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --link . .
ENTRYPOINT [ "python", "chat.py" ]