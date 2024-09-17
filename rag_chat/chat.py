"""Simple demo chat application using LLMs via ollama"""

import asyncio
import signal
import sys


from typing import List

from ollama import AsyncClient, Message

from db import db


SHOULD_EXIT = False
OLLAMA_HOST = 'http://localhost:11434'
EMBEDDINGS_MODEL = "mxbai-embed-large"

ollama_client = AsyncClient(host=OLLAMA_HOST)


async def main() -> None:
    """Entrypoint to the application"""

    print("\n--- Demo RAG chat app with ollama ---\n")

    # get the model to use from the first command line argument
    # if not provided, use the default model
    chat_model = sys.argv[1] if len(sys.argv) > 1 else "gemma2:2b"

    # pull models so we're sure they're available
    await pull_models([chat_model, EMBEDDINGS_MODEL])

    # pick whatever model you want to chat with
    # some may behave oddly or expect messages to be formatted differently
    # HAVE FUN!
    await chat(chat_model)


async def pull_model(model: str):
    """uses ollama to pull a model"""

    if await is_model_installed(model):
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


async def pull_models(models: List[str]):
    """uses ollama to pull a list of models"""

    for model in models:
        await pull_model(model)


async def is_model_installed(model: str) -> bool:
    """checks if a model is installed in ollama"""

    installed_models: List[str] = []

    if (ollama_models := ((await ollama_client.list()).get("models"))) is not None:
        installed_models = [model.get("name") for model in ollama_models]

    return model in installed_models


async def chat(model: str):
    """starts the demo chat app"""

    print(f"Starting a chat with {model}.\n")

    chat_thread: List[Message] = [
        Message(
            role="system",
            content="""
                You are an expert assistant. Your job is to help the user answer their questions.
                Be as helpful as you possibly can. Be cool. Be suave. Very demure.
                You can only help with Docker, Docker products, and Docker-related topics and questions. For anything else, respond that it's outside your area of expertise. 
                Give thorough responses with examples
                Use the provided <context></context> as your main source of information when responding,
                and always reference the source you use in your response.
            """
        ),
        Message(
            role="assistant",
            content="""
                You are an assistant, attempting to help the user as best you can. Below is the entire chat thread.
                When responding with code, use markdown. Format your responses in markdown whenever appropriate.
            """
        )
    ]

    # main chat loop. quit by typing "exit" or by hitting CTRL-C
    while not SHOULD_EXIT:
        user_input: str = input('>> ')

        if user_input.lower() == 'exit':
            break

        # TODO: create an embedding for the query, search the vector db for similarities,
        # and add the text of the closest results to the chat thread as context for a better response

        # 1) generate embedding
        embedding_res = await ollama_client.embed(model=EMBEDDINGS_MODEL, input=user_input)
        if not embedding_res:
            print("No embeddings found for the input.")
            continue
        input_embeddings = embedding_res.get("embeddings")
        if input_embeddings is not None:
            input_embeddings = input_embeddings[0]
        else:
            input_embeddings = []
        
        # 2) query db for context chunks and their source urls
        res = await db.get_nearest_neighbors(embedding=input_embeddings, limit=5)
        context_msg = \
'''
Below, between the <context></context> tags, are some relevant chunks of 
text from the documentation that might help you answer the user's question. 
Remember to always include the source of information in your response.

<context>
'''
        for record in res:
            msg_content = f"Source URL: {record["source_url"]}\n\n{record["chunk_text"]}"
            context_msg += f"---\n\n{msg_content}\n\n"

        context_msg += "</context>"

        # 3) add the context to the chat thread, either:
        #    - as part of the user message (most likely), or
        #    - as a separate message
        chat_thread.append(Message(role="user", content=context_msg + f"\n\nUser query: {user_input}"))

        # 4) chat with the model
        response = await ollama_client.chat(
            model=model,
            messages=chat_thread,
            stream=True,
            options={'temperature': 0.85}
        )

        llm_response_msg = ''
        async for part in response:
            if msg := part.get("message"):
                llm_response_msg += msg.get("content")
                print(msg.get("content"), sep='', end='', flush=True)

        # add the llms response to the thread so it keeps context after the next question
        chat_thread.append(Message(role="assistant", content=llm_response_msg))

        print("\n")


def sigint_handler(sig, frame):
    """handles sigint signals, used only to exit the application"""
    global SHOULD_EXIT
    print('\n\nExiting demo chat... Goodbye!')
    SHOULD_EXIT = True
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    asyncio.get_event_loop().run_until_complete(main())
