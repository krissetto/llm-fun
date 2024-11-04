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

from ollama import AsyncClient, Message

import db


# number of recent messages to keep in the chat history when generating a search query
HISTORY_CUTOFF = 5


async def create_embeddings(chunked_docs: Dict[str, List[str]]) -> None:

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
            vectors = (
                await ollama_client.embed(model=EMBEDDINGS_MODEL, input=f"search_document: {chunk}")
            ).get('embeddings')
            if vectors:
                embeddings.append((chunk, vectors))
        if embeddings:
            await db.save_chunked_embeddings({url: embeddings})


async def generate_content_embeddings_by_semantics(
        chunked_docs: Dict[str, List[str]],
        ai_client: AsyncClient, 
        semantic_model: str,
        embeddings_model: str
) -> None:
    '''TODO: consolidate with [create_embeddings]'''
    

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
            
            # generate semantic meaning of chunk, and create embeddings
            # of that instead of the chunk text itself.
            #
            # the idea is to make the embeddings more easily searchable by similarity,
            # since we'll also be interpreting the users questions in
            # the same way to try and make the two sides (input question embeddings
            # and document chunk embeddings) more closely aligned with each other

            semantic_meaning_prompt = f"""
You are an AI language model tasked with summarizing content for search indexing. Generate a concise summary that captures the main ideas and concepts of the text. Follow these guidelines:

- The summary must be easily searchable and serve as an index for retrieving the full content later.
- Avoid copying the text verbatim.
- Always include code snippets, commands, error examples, or any information that is often included in searches.
- Exclude URLs, links, and step-by-step instructions.
- Focus on explanations of the content and possible search-related keywords.
- Keep the summary under 100 words where possible.
- Respond ONLY with the summary text. Do not include any other information, headings, titles, or anything else.

Content:
{chunk}
"""
            
#             You are an AI language model tasked with understanding the content and generating a text description of the content for use as a search index. Generate a concise and clear summary that captures the main ideas and concepts of the text. Avoid copying the text verbatim. You must follow these guidelines:

# - The summary much be easily searchable and serve as an index for retrieving the full content later. 
# - Do not answer in a chat like manner. 
# - Include code snippets, commands, or any other relevant information that may be included in a user search query.
# - Do not include any URLs or links.
# - Do not include any step by step instructions or content needed to answer the query, only what's necessary to find the content.
# - Explanations of the substance of the content and possible search related keywords are preferred over 'do this and that' guides (e.g. "main concepts of 'x'", "explanation of 'y' keywords", etc.).
# - Keep the summary under 100 words where possible.
# - Answer ONLY with the text of your summary. Do not include any other information.

# Content:
# {chunk}
            semantic_meaning = (
                await ai_client.generate(
                    model=semantic_model,
                    prompt=semantic_meaning_prompt,
                    options={'temperature': 0.35, 'num_ctx': 8192}
                )
            ).get('response')

            semantic_vectors = (
                await ai_client.embed(model=embeddings_model, input=f"search_document: {semantic_meaning}")
            ).get('embeddings')

            if semantic_vectors:
                embeddings.append((chunk, semantic_vectors))

        if embeddings:
            await db.save_chunked_embeddings({url: embeddings})
    


async def generate_search_embeddings(
        chat_thread: List[Message], 
        user_input: str, 
        ollama_client: AsyncClient, 
        model: str
) -> List:
    if len(chat_thread) < 2:
        # this is when the user sent their first message in the thread
        embedding_res = await ollama_client.embed(model=EMBEDDINGS_MODEL, input=f"search_query: {user_input}")
    else:
        # this chat has a history, let's use it to best determine the search query when querying the vector db
        # use the chat history to generate the search query using the main llm model
        # remove irrelevant system and assistant prompts (at least the first 2 messages)
        msgs_for_search_query = chat_thread[2:]
        msgs_for_search_query = msgs_for_search_query[-HISTORY_CUTOFF:]
        
        rag_context_prompt = """
    You are a search engine query generator. Based on the message history provided, create a search query for the user's last message. This search query will be used to gather relevant documents to better answer the user's question.

    Examples:

    If in a previous message the user asked 'what is x?' and they now ask 'how can I use it?', respond with 'using x', clarifying what 'it' stands for and writing it in a way to optimize for finding relevant documents.

    <chat_history>


    """

        rag_context_prompt += ""
        for msg in msgs_for_search_query:
            rag_context_prompt += f"---\n\nrole: {msg.get('role')}\nmessage: {msg.get('content')}\n\n"
        rag_context_prompt += "</chat_history>"
        rag_context_prompt += f"\n<last_user_message>{user_input}</last_user_message>\n\n"

        rag_context_prompt += """
Answer by rewriting the user's last message as a search query for a search engine. Do not explain your answer, just output the search query to use. The search query is always a single short question or statement that represents the areas of knowledge required for answering the user's question. You will never answer the user's question directly.
"""

        search_query_res = await ollama_client.generate(
            model=model,
            prompt=rag_context_prompt,
            options={'temperature': 0.35, 'num_ctx': 16384}
        )

        print(f"\n---\nUsing the following adapted search query: \"{search_query_res.get('response')}\"\n---\n")

        embedding_res = await ollama_client.embed(model=EMBEDDINGS_MODEL, input=f"search_query: {search_query_res.get('response')}")
        

    if not embedding_res:
        print("Error creating embedding for the user's input")
        return []
    input_embeddings = embedding_res.get("embeddings")
    if input_embeddings is not None:
        input_embeddings = input_embeddings[0]
    else:
        input_embeddings = []

    return input_embeddings


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
