'''funcs for db ops'''

from datetime import datetime
from typing import Dict, List, Tuple

from . import db_conn
from .queries import (
    GET_ALL_URLS, INSERT_MULTIPLE_URLS,
    REMOVE_URL_FROM_SITEMAP, UPDATE_LASTMOD,
    INSERT_CHUNKED_EMBEDDING_QUERY, GET_SIMILAR_EMBEDDINGS,
    INSERT_URL_MAPPING, INSERT_CHUNK_TEXT_MAPPING
)


async def save_sitemap(sitemap: dict[str,datetime]) -> dict[str,datetime]:
    '''
        Checks, naively, if any new url from the downloaded sitemap needs to be added to the db.
        Returns a dict containing only the updated or newly added entries.
    '''
    current_entries = await db_conn.fetch(GET_ALL_URLS)

    if len(current_entries) == 0:
        await db_conn.executemany(INSERT_MULTIPLE_URLS, [(url, lastmod) for url, lastmod in sitemap.items()])
        return sitemap

    urls_in_db: dict[str, datetime] = {record["url"]: record["lastmod"] for record in current_entries}

    records_to_insert = {u: l for u, l in sitemap.items() if urls_in_db.get(u) is None}
    records_to_update: dict[str, datetime] = {}
    records_to_remove: list[str] = []

    for url, lastmod in urls_in_db.items():
        if (new_url_lastmod := sitemap.get(url)) is None:
            records_to_remove.append(url)
        elif new_url_lastmod > lastmod:
            records_to_update[url] = sitemap[url]

    print(f"DB: {len(records_to_insert)} records need to be inserted")
    await db_conn.executemany(INSERT_MULTIPLE_URLS, [(url, lastmod) for url, lastmod in records_to_insert.items()])

    print(f"DB: {len(records_to_update)} records need to be updated")
    await db_conn.executemany(UPDATE_LASTMOD, [(lastmod, url) for url, lastmod in records_to_update.items()])

    print(f"DB: {len(records_to_remove)} records need to be deleted")
    await db_conn.executemany(REMOVE_URL_FROM_SITEMAP, [(url,) for url in records_to_remove])

    return {**records_to_insert, **records_to_update}


async def save_chunked_embeddings(chunked_embeddings: Dict[str, List[Tuple[str, List[float]]]]):
    '''
        Saves the chunked embeddings to the db
    '''
    for url, data in chunked_embeddings.items():
        # Insert URL into url_mapping table and get the URL ID
        url_id = await db_conn.fetchval(INSERT_URL_MAPPING, url)
        if url_id is None:
            url_id = await db_conn.fetchval('select id from url_mapping where url = $1', url)

        for chunk_text, embedding in data:
            # Insert chunk text into chunk_text_mapping table and get the chunk text ID
            chunk_text_id = await db_conn.fetchval(INSERT_CHUNK_TEXT_MAPPING, chunk_text)
            if chunk_text_id is None:
                chunk_text_id = await db_conn.fetchval('select id from chunk_text_mapping where chunk_text = $1', chunk_text)

            # Insert into embeddings table using the URL ID and chunk text ID
            await db_conn.execute(
                INSERT_CHUNKED_EMBEDDING_QUERY,
                url, chunk_text, embedding[0]
            )


# async def save_chunked_embeddings(chunked_embeddings: Dict[str, List[Tuple[str, List[float]]]]):
#     '''
#         Saves the chunked embeddings to the db
#     '''
#     for url, data in chunked_embeddings.items():
#         await db_conn.executemany(
#             INSERT_CHUNKED_EMBEDDING_QUERY,
#             # TODO: double check this embedding list
#             [(url, text, embedding[0]) for text, embedding in data]
#         )


async def get_nearest_neighbors(embedding: List[float], limit: int) -> List[Tuple[str,str]]:
    '''
        Gets the nearest neighbors to the given embedding (the closest result of the vector search)

        Returns a list of (source_url, chunk_text) tuples, for use in a RAG prompt
    '''
    return await db_conn.fetch(GET_SIMILAR_EMBEDDINGS, embedding, limit)
