"""Simple demo chat application using LLMs via ollama"""

import asyncio
import os
import signal
import sys


from typing import List

from ollama import AsyncClient, Message


SHOULD_EXIT = False
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", 'http://localhost:11434')

ollama_client = AsyncClient(host=OLLAMA_HOST)


async def main() -> None:
    """Entrypoint to the application"""

    print(f"\n--- Demo LLM chat app with ollama (host: {OLLAMA_HOST}) ---\n")

    # get the model to use from the first command line argument
    # if not provided, use the default model
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma2:2b"

    # pull model so we're sure it's available
    await pull_model(model)

    # pick whatever model from the available ones listed above.
    # some may behave oddly or expect messages to be formatted differently
    # HAVE FUN!
    await chat(model)


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
                You are a drunken sailor.
                Be as helpful as you possibly can, never letting your personality go unnoticed, but also
                never being too on the nose with your jokes etc. Be cool. Be suave. JARR
            """
        ),
        Message(
            role="assistant",
            content="""
                You are an assistant, attempting to help the user as best you can. Below is your entire chat thread.
                When responding with code, use markdown. Format your responses nicely.
            """
        )
    ]

    # main chat loop. quit by typing "exit" or by hitting CTRL-C
    while not SHOULD_EXIT:
        user_input: str = input('>> ')

        if user_input.lower() == 'exit':
            break

        chat_thread.append(Message(role="user", content=user_input))

        response = await ollama_client.chat(
            model=model,
            messages=chat_thread,
            stream=True,
            options={'temperature': 0.85}
        )

        llm_response_msg = ''
        async for part in response:
            if msg := part.get("message", None):
                llm_response_msg += msg.get("content", None)
                print(msg.get("content", None), sep='', end='', flush=True)

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
    asyncio.run(main())
