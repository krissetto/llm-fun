'''super fugly db initialization stuff because i'm lazy'''

import asyncio
import asyncpg

from pgvector.asyncpg import register_vector


db_conn: asyncpg.connection.Connection = None


async def _init():
    global db_conn
    if db_conn is not None:
        raise RuntimeError("DB has already been created, something weird is happening")

    db_conn = await asyncpg.connect(user="postgres", password="postgres", database="rag", host="localhost", port=6024)
    await db_conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    await register_vector(db_conn)
    
    await db_conn.execute("""
create table if not exists sitemap (
    id int generated by default as identity primary key,
    url text not null,
    lastmod timestamp
);

create table if not exists url_mapping (
    id int generated by default as identity primary key,
    url text not null unique
);

create table if not exists chunk_text_mapping (
    id int generated by default as identity primary key,
    chunk_text text not null,
    chunk_text_hash text generated always as (md5(chunk_text)) stored,
    unique (chunk_text_hash)
);

-- Create the embeddings table to reference the mapping tables
create table if not exists embeddings (
    id int generated by default as identity primary key,
    url_id int not null references url_mapping(id),
    chunk_text_id int not null references chunk_text_mapping(id),
    chunk_embedding vector(1024),
    unique (url_id, chunk_text_id)
);
""")
                          
    
#     await db_conn.execute("""
# create table if not exists sitemap (
#     id int generated by default as identity primary key,
#     url text not null,
#     lastmod timestamp
# );

# create table if not exists embeddings (
#     id int generated by default as identity primary key,
#     source_url text not null,
#     chunk_text text not null,
#     chunk_embedding vector(1024),
#     unique (source_url, chunk_text)
# );""")


asyncio.get_event_loop().run_until_complete(_init())

# This is here at the bottom to import the funcs from db.py to make them
# easier to use by only writing `import db` in the file that needs them.
# Those funcs use the db_conn pkg global, so putting this line at the top causes a circular import error
from .db import *
