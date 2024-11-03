'''
    Basic features:
    - Create embeddings for docker documentation
    - Store said embeddings in sqlite vector database
    - Be able to query the embeddings with natural text
    - Reply to query with coherent answer using LLM, and surfacing the sources used + links to the relevant docs

    Goals: 
    - Make querying the documentation fast and lightweight enough for local use on sufficiently modern pcs
    - Be able to update the docs embeddings behind the scenes when there are updates
'''

import os

from typing import Dict, List, Tuple

from ollama import AsyncClient

import db

from chunking.chunking import chunk_docs
from scraping.scraping import get_docs_to_embed

# EMBEDDINGS_MODEL = "mxbai-embed-large"
EMBEDDINGS_MODEL = "nomic-embed-text"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", 'http://localhost:11434')


async def create_embeddings() -> None:
    '''
        To-Do:
        - Get docs [X]
        - Split docs into chunks
        - Create vector embeddings for the chunks
        - Store vector embeddings into vector sqlite database
    '''
    # (url, content) dict
    docs = await get_docs_to_embed()
    if len(docs) == 0:
        print("All docs are updated, there is nothing new to embed")
        return

    # key is the url, value is a list of chunks for each url
    chunked_docs = chunk_docs(docs, chunk_size=10, chunk_overlap=2)

    ollama_client = AsyncClient(host=OLLAMA_HOST)

    await pull_model(ollama_client, EMBEDDINGS_MODEL)

    # create embeddings from chunked docs and save to db
    num_urls = len(chunked_docs.keys())
    url_index = 0
    for url, url_content_chunks in chunked_docs.items():
        # TODO: clean all this shit up
        # this type is a dict -> url: (chunk_text, chunk_embedding_vector)
        url_index += 1
        num_chunks = len(url_content_chunks)
        embeddings: List[Tuple[str,List[float]]] = []
        # TODO(krissetto): make this better, e.g. by making a single call
        # to ollama for all the chunks of a url
        for i, chunk in enumerate(url_content_chunks):
            print(f"embedding chunk {i} of url {url}\n({url_index}/{num_urls} urls | {i}/{num_chunks} chunks)\n\n")
            vectors = (await ollama_client.embed(model=EMBEDDINGS_MODEL, input=f"search_document: {chunk}")).get('embeddings')
            if vectors:
                embeddings.append((chunk, vectors))
        if embeddings:
            await db.save_chunked_embeddings({url: embeddings})


async def pull_model(ollama_client: AsyncClient, model: str):
    """uses ollama to pull a model"""

    if await is_model_installed(ollama_client, model):
        print(f"{model} is already installed.\n")
        return

    print(f"pulling {model}...\n")
    progress = await ollama_client.pull(model=model, stream=True)

    async for part in progress:
        status = part.get('status', None)

        if (total := part.get('total', None)) and (completed := part.get('completed', None)):
            perc_complete = (completed*100) / total
            print(f"model: {model} - {status} - completed: {perc_complete:.2f}%")
        else:
            print(f"model: {model} - {status}")

    print("\n\n")


async def is_model_installed(ollama_client: AsyncClient, model: str) -> bool:
    """checks if a model is installed in ollama"""

    installed_models: List[str] = []

    if (ollama_models := ((await ollama_client.list()).get("models"))) is not None:
        installed_models = [model.get("name") for model in ollama_models]

    return model in installed_models
