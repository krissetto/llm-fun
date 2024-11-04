'''the most beautiful of db queries'''

GET_ALL_URLS = '''
select url, lastmod from sitemap
'''

INSERT_MULTIPLE_URLS = '''
insert into sitemap (url, lastmod) values ($1, $2)
'''

UPDATE_LASTMOD = '''
update  sitemap
set     lastmod = $1
where   url = $2;
'''

REMOVE_URL_FROM_SITEMAP = '''
delete  
from    sitemap
where   url = $1;
'''


# Insert URL and chunk text into their respective mapping tables
INSERT_URL_MAPPING = '''
insert into url_mapping (url) values ($1)
on conflict (url) do nothing
returning id
'''


INSERT_CHUNK_TEXT_MAPPING = '''
insert into chunk_text_mapping (chunk_text) values ($1)
on conflict (chunk_text_hash) do nothing
returning id
'''


INSERT_CHUNKED_EMBEDDING_QUERY = '''
insert into embeddings (url_id, chunk_text_id, chunk_embedding) 
values (
    (select id from url_mapping where url = $1), 
    (select id from chunk_text_mapping where chunk_text_hash = md5($2)), 
    $3
)
on conflict (url_id, chunk_text_id)
do update set chunk_text_id = (select id from chunk_text_mapping where chunk_text_hash = md5($2)), chunk_embedding = $3;
'''


GET_SIMILAR_EMBEDDINGS = '''
select 
    url_mapping.url as source_url, 
    chunk_text_mapping.chunk_text, 
    1 - (embeddings.chunk_embedding <-> $1) as cosine_similarity
from embeddings
join url_mapping on embeddings.url_id = url_mapping.id
join chunk_text_mapping on embeddings.chunk_text_id = chunk_text_mapping.id
order by cosine_similarity desc
limit $2
'''