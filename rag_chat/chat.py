"""Simple demo chat application using LLMs via ollama"""

import asyncio
import os
import signal
import sys


from typing import List

from ollama import AsyncClient, Message

from db import db

from embedding.embedding import generate_embeddings, generate_search_embeddings


SHOULD_EXIT = False
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", 'http://localhost:11434')
EMBEDDINGS_MODEL = "nomic-embed-text"

ollama_client = AsyncClient(host=OLLAMA_HOST)


async def main() -> None:
    """Entrypoint to the application"""

    print("\n--- Demo RAG chat app with ollama ---\n")

    # get the model to use from the first command line argument
    # if not provided, use the default model
    chat_model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"

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
You are an AI assistant specialized in Docker and Docker-related technologies. Your knowledge encompasses the latest information about Docker tools, including Docker Build Cloud, Docker Scout, Docker Debug, and other Docker ecosystem products.

Your goal is to assist users with Docker-related queries and tasks. Adhere to these rules:

1. **Focus on Docker**: Address only Docker-related topics. Politely inform users if a question is outside this scope.
2. **Provide Accurate Information**: Ensure responses are accurate and up-to-date, reflecting the latest Docker technologies and best practices.
3. **Be Clear and Concise**: Offer clear, concise, and practical responses. Avoid unnecessary tangents unless specifically requested. When providing examples, limit them to at most 2-3 and keep them short.
4. **Consider Context**: Pay attention to the most recent messages in the conversation history to maintain context.
5. **Explain Commands and Configurations**: Provide clear but brief examples.
6. **Acknowledge Uncertainty**: If unsure, acknowledge it and suggest where to find more accurate information.
7. **Generate Accurate Code**: Follow best practices when creating Docker-related code. Avoid unnecessary changes the user hasn't requested.
8. **Use Provided Context**: Rely only on the provided context for information. Ask for more details if needed.
9. **Avoid Hallucination**: Do not make up information. To provide accurate information, used the context provided
10. **Cite Sources**: Always cite your sources, including URLs, at the end of each message.

Your objective is to be a helpful, accurate, brief and user-friendly assistant for all Docker-related inquiries."""
        ),
#         Message(
#             role="assistant",
#             content="""
# You are an assistant, attempting to help the user as best you can. 
# You have access to the most recent chat messages in the thread, keep them into consideration.
# When responding with code, use markdown. Format your responses in markdown whenever appropriate.
#             """
#         )
    ]

    # main chat loop. quit by typing "exit" or by hitting CTRL-C
    while not SHOULD_EXIT:
        user_input: str = input('>> ')

        if user_input.lower() == 'exit':
            break

#         # 1) generate embedding for question
#         input_embeddings = await generate_search_embeddings(
#             chat_thread=chat_thread, 
#             user_input=user_input, 
#             ai_client=ollama_client, 
#             llm_model=model, 
#             embeddings_model=EMBEDDINGS_MODEL
# )

#         # 2) query db for context chunks and their source urls
#         res = await db.get_nearest_neighbors(embedding=input_embeddings, limit=5)

        # 1) generate embedding for question
        input_embeddings = await generate_embeddings(
            user_input=user_input,
            ai_client=ollama_client,
            embeddings_model=EMBEDDINGS_MODEL)

        # 2) query db for context chunks and their source urls
        res = await db.get_nearest_neighbors(embedding=input_embeddings, limit=5)

        context_msg = '''
<context>

'''
# Remember to always include the source of the information you use in your response.
# If the context doesn't seem relevant to the user's message, ignore it completely.
# If you don't seem to have enough context for answering, let the user know.

        for record in res:
            msg_content = f"Source URL: {record['source_url']}\nContents:\n\n{record['chunk_text']}"
            context_msg += f"---\n\n{msg_content}\n\n"

        context_msg += "</context>"

        print(f"Context:\n{context_msg}\n")

        # 3) add the context to the chat thread, either:
        #    - as part of the user message (most likely), or
        #    - as a separate message
        chat_thread.append(Message(role="system", content=context_msg))
        chat_thread.append(Message(role="user", content=user_input))

        # 4) chat with the model
        response = await ollama_client.chat(
            model=model,
            messages=chat_thread,
            stream=True,
            options={'temperature': 0.35, 'num_ctx': 8192, 'num_thread': 12}
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
