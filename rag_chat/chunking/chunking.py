'''for chunking stuff'''

from time import perf_counter
from typing import Dict, List

from nltk import tokenize, download as nltk_download
from ollama import AsyncClient


def chunk_docs(docs: Dict[str, str], chunk_size: int, chunk_overlap: int) -> Dict[str, List[str]]:
    '''
        Chunks docs into smaller pieces for better embeddings
        
        Returns a dict with the url as a key and a list of chunks as the value
    '''
    output = {}

    for url, url_contents in docs.items():
        chunked_content = _chunk_by_sentence(
            url_contents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        for i, sentences in enumerate(chunked_content):
            chunked_content[i] = " ".join(sentences)
        output[url] = chunked_content

    return output


async def chunk_docs_semantically(
        docs: Dict[str, str],
        ai_client: AsyncClient, 
        llm_model: str
) -> Dict[str, List[str]]:
    '''
        Chunks docs into smaller pieces for better embeddings, using an llm to determine the chunks (?)
        
        Returns a dict with the url as a key and a list of chunks as the value
    '''
    output = {}

    start_time = perf_counter()

    for url, url_contents in docs.items():
        chunked_content = await _chunk_by_semantics(
            content=url_contents,
            ai_client=ai_client,
            llm_model=llm_model
        )
        output[url] = chunked_content
    
    print(f"Chunking all documents took {perf_counter() - start_time:0.4f} seconds")

    return output


async def _chunk_by_semantics(content: str, ai_client: AsyncClient, llm_model: str) -> List[str]:
    '''
        Chunks content semantically, according to what the llm thinks ;)
        
        Returns a list of chunks as strings
    '''
    chunking_prompt = f'''
You are an AI documentations chunker. Your task is to divide markdown documentation into smaller, more manageable chunks.
You must ensure that the chunks are coherent and that they make sense on their own.
You must also remove any irrelevant information such as navigation links, footers, etc.
Your response chunks must be separated using the "---" sequence (three "-" chars) as a separator.
Chunk contents must be as close to the original text as possible.
Each chunk should be roughly 500 words long at most. You can make exceptions for long code blocks.
NEVER split a code block, a command, or generally anything inside a `` or ``` ``` markdown block
Do not include any text other than the chunks and the separator, and do not add whitespace.
Do not add anything to the text you are given.

The text you have to chunk is the following:

{content}
'''
    start_time = perf_counter()
    chunking_res: str = (await ai_client.generate(
        model=llm_model,
        prompt=chunking_prompt,
        options={'temperature': 0.35, 'num_ctx': 8192, 'num_thread': 12}
    )).get('response', "")

    # split the response into chunks on the "---" separator, trimming whitespace, excluding empty chunks
    chunks : List[str] = []
    chunks += [chunk.strip() for chunk in chunking_res.split("---") if chunk.strip()]

    print(f"Chunking content into {len(chunks)} pieces took {perf_counter() - start_time:0.4f} seconds")

    return chunks


def _chunk_by_sentence(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    '''
        Chunks content by sentence.
        
        Parameters:
        - chunk_size: number of sentences to have in each chunk
        - chunk_overlap: number of sentences that overlap between adjacent chunks

        Returns:
        - tuple with the content as the first value and the list of chunks as the second value
    '''
    nltk_download('punkt_tab')
    tokenized_content = tokenize.sent_tokenize(content)

    tc_length = len(tokenized_content)
    chunks: List[str] = []
    chunk_index = 0
    next_index = chunk_size
    while chunk_index < tc_length:
        if chunk_index > 0:
            next_index = chunk_index + chunk_size
            if next_index > tc_length:
                next_index = tc_length-1
        chunks.append(
            tokenized_content[chunk_index:(chunk_index + chunk_size)]
        )
        chunk_index += chunk_size - chunk_overlap


    return chunks

def _chunk_by_word(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    '''
        Chunks content by word.
        
        Parameters:
        - chunk_size: number of words to have in each chunk
        - chunk_overlap: number of words that overlap between adjacent chunks

        Returns:
        - tuple with the content as the first value and the list of chunks as the second value
    '''
    nltk_download('punkt_tab')
    tokenized_content = tokenize.word_tokenize(content)

    tc_length = len(tokenized_content)
    chunks: List[str] = []
    chunk_index = 0
    next_index = chunk_size
    while chunk_index < tc_length:
        if chunk_index > 0:
            next_index = chunk_index + chunk_size
            if next_index > tc_length:
                next_index = tc_length-1
        chunks.append(
            tokenized_content[chunk_index:(chunk_index + chunk_size)]
        )
        chunk_index += chunk_size - chunk_overlap


    return chunks
