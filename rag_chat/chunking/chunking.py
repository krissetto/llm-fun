'''for chunking stuff'''

from typing import Dict, List

from nltk import tokenize, download as nltk_download


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

def _chunk_by_semantics(content: str, chunk_overlap: int) -> List[str]:
    '''
        Chunks content by semantic elements, according to what the llm thinks
        
        Returns a list of chunks as strings
    '''
    raise NotImplementedError


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
