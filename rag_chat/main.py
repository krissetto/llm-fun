'''
    Basic features:
    - Create embeddings for docker documentation
    - Store said embeddings in sqlite vector database
    - Be able to query the embeddings with natural text
    - Reply to query with coherent answer using LLM, and surfacing the sources used + links to the relevant docs

    Goals: 
    - Make querying the documentation fast and lightweight enough for local use on sufficiently modern pcs
    - Be able to update the docs embeddings behind the scenes when there are updates

    To-Do:
        - [x] Get user query
        - [x] Create embedding from query
        - [x] Semantic search vector db with query embedding
        - [x] Get top n results
        - [x] Craft custom prompt for LLM with results, based on some template
        - [x] Query LLM with custom prompt
        - [x] Add the references to the LLM's output
        - [x] Give user the final response
        - (bonus) save user question and answer for stats etc.
'''


import sys
import os

import asyncio

from ollama import AsyncClient

from chunking.chunking import chunk_docs_semantically
from embedding.embedding import generate_content_embeddings_by_semantics, pull_model
from scraping.scraping import get_docs_to_embed


EMBEDDINGS_MODEL = "nomic-embed-text"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", 'http://localhost:11434')


async def main():
    '''de-facto entrypoint'''
    # (url, content) dict
    docs = await get_docs_to_embed()
    if len(docs) == 0:
        print("All docs are updated, there is nothing new to embed")
        return
    

    # get the model to use from the first command line argument
    # if not provided, use the default model
    chat_model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"
    print("using chat model: ", chat_model)

    ollama_client = AsyncClient(host=OLLAMA_HOST)
    await pull_model(ollama_client, EMBEDDINGS_MODEL)
    await pull_model(ollama_client, chat_model)
    
    # key is the url, value is a list of chunks for each url
    chunked_docs = await chunk_docs_semantically(docs, ollama_client, chat_model)

    # await create_embeddings(chunked_docs) <- this is the old way, will need adapting
    await generate_content_embeddings_by_semantics(chunked_docs, ollama_client, chat_model, EMBEDDINGS_MODEL)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
