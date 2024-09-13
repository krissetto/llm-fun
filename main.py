import asyncio
import signal
import sys


from typing import List

from ollama import AsyncClient, Message


ollama_host = 'http://localhost:11434'
ollama_client = AsyncClient(host=ollama_host)

should_exit = False


async def main():
    models: List[str] = ["llama3.1:8b", "phi3.5:3.8b", "gemma2:2b", "qwen2:0.5b"]
    
    await pull_models(models) # pull models so we're sure they are available
    
    # pick whatever model from the available ones listed above.
    # some may behave oddly or expect messages to be formatted differently
    # HAVE FUN!
    await chat("gemma2:2b")
    
    
async def pull_models(models: List[str]):
    
    for model in models:
        
        print(f"pulling {model}...\n")
        progress = await ollama_client.pull(model=model, stream=True)
        
        async for part in progress:
            status = part.get('status', None)
            
            if (total := part.get('total', None)) and (completed := part.get('completed', None)):
                perc_complete = (completed*100) / total
                print(f"model: {model} - {status} - completed: {perc_complete:.2f}%")
            else:
                print(f"model: {model} - {status}")
        
        print("\n")
    
    print(f"All models have been pulled! ({' '.join(models)})\n\n")


async def chat(model: str):
    print("--- CHAT TIME ---\n\n")

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

    while not should_exit:
        user_input: str = input('>> ')

        if user_input.lower() == 'exit':
            break

        chat_thread.append(Message(role="user", content=user_input))

        print("\n---\n\n")

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

        print("\n\n")


def signal_handler(sig, frame):
    global should_exit
    print('Exiting...')
    should_exit = True
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(main())