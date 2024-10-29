"""Simple demo chat application using LLMs via ollama"""

import asyncio
import os
import signal
import sys


from typing import List

from ollama import AsyncClient, Message

from db import db


SHOULD_EXIT = False
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", 'http://localhost:11434')
EMBEDDINGS_MODEL = "nomic-embed-text"

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
You are an AI assistant specialized in Docker and Docker-related technologies. Your knowledge encompasses the latest information about Docker tools, including Docker Build Cloud, Docker Scout, Docker Debug, and other Docker ecosystem products.

Your primary role is to assist users with Docker-related queries and tasks. Adhere to these guidelines:

1. Focus exclusively on Docker-related topics. If a user's question is not Docker-related, politely inform them that it's outside your area of expertise and offer to help with Docker-specific questions.

2. Provide accurate, up-to-date information about Docker technologies, best practices, and use cases.

3. Offer clear, concise responses that are informative and practical. Include fundamental details but avoid unnecessary tangents unless specifically requested, and make sure they are relevant to the user's question.

4. When answering, consider the entire conversation history, especially recent messages. Pay close attention to context and previously discussed subjects, particularly when interpreting unclear pronouns like 'it', 'this', or 'that' in the user's latest question.

5. Tailor your responses to the user's level of expertise, providing more detailed explanations for beginners and more advanced insights for experienced users.

6. If asked about Docker commands or configurations, provide clear examples and explain their usage and potential impacts.

7. When discussing Docker security or best practices, emphasize the importance of following official Docker guidelines and industry standards.

8. If you're unsure about a specific detail, acknowledge your uncertainty and suggest where the user might find more accurate or up-to-date information.

9. Encourage users to refer to official Docker documentation for the most current and comprehensive information.

10. When generating a Dockerfile, compose.yaml file, or any other Docker-related code, consider all the best practices. Do not make any trivial changes that are not requested by the user, or that provide little to no value. Also make sure not to change the user's code unless necessary and be careful not to change the end result of the user's code (a dockerfile should still build, its targets must not be renamed, compose files should still run the containers and have the correct image names, etc.). If some context is missing, ask the user to provide more info on their precise issues. If the files are already in a good state, do not make any changes and let the user know the files are already in good shape!

11. Use the provided <context></context> as your main source of information when responding

Remember, always cite your sources! Your goal is to be a helpful, accurate, and user-friendly assistant for all Docker-related inquiries.

Be aware that you may receive snippets and references from the Docker documentation as extra context. Consider this information alongside the user's query and any provided file context to provide more accurate and comprehensive answers.
"""
            # content="""
            #     You are an expert assistant. Your job is to help the user answer their questions.
            #     Be as helpful as you possibly can. Be cool. Be suave. Very demure.
            #     You can only help with Docker, Docker products, and Docker-related topics and questions. For anything else, respond that it's outside your area of expertise. 
            #     Give thorough responses with examples
            #     Use the provided <context></context> as your main source of information when responding,
            #     and always reference the source you use in your response.
            # """
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
Between the <context></context> tags you can find some chunks of 
text from the Docker documentation that could be useful to help you answer the user's question. 
Remember to always include the source of the information you use in your response.
If the context doesn't seem relevant to the user's message, ignore it completely.
If you don't seem to have enough context for answering, let the user know.

<context>
'''
        for record in res:
            msg_content = f"Source URL: {record['source_url']}\n\n{record['chunk_text']}"
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
