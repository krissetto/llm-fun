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


import asyncio

from chunking.chunking import chunk_docs
from embedding.embedding import create_embeddings
from scraping.scraping import get_docs_to_embed


async def main():
    '''de-facto entrypoint'''
    # (url, content) dict
    docs = await get_docs_to_embed()
    if len(docs) == 0:
        print("All docs are updated, there is nothing new to embed")
        return
    
    # key is the url, value is a list of chunks for each url
    chunked_docs = chunk_docs(docs, chunk_size=10, chunk_overlap=2)
    
    await create_embeddings(chunked_docs)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
