# Running the infrastructure for the basic chat

Run the compose file without specifying any profiles to launch the default services: `docker compose up -d`.  

This will spin up:

- An `Ollama` server (exposed on port `11434`);
- A `Open WebUI` instance for a chatGPT-like experience to play around with (exposed on port `3000`);

You should now be able to visit `localhost:3000` and create your own local openwebui account

## Stopping the basic chat infra

To stop the basic chat infra, run: `docker compose down`

# Running the basic chat demo

You must have a somewhat modern version of python and Docker installed

- (**If not already present**) Create a venv in your current dir: `python3 -m venv venv`
- (**If not already activated**) Activate the venv: `source venv/bin/activate`
- Install deps: `pip install -r basic_chat/requirements.txt`
- Run the demo chat app: `python3 basic_chat/chat.py gemma2:2b`  
  (replace `gemma2:2b` with any model present on [ollama](https://ollama.com/library/))


# Running the infrastructure for the RAG chat

Run the compose file with the `rag` profile: `docker compose --profile rag up -d`.  

This will spin up:

- An `Ollama` server (exposed on port `11434`);
- A `Open WebUI` instance, for a chatGPT-like experience to play around with (exposed on port `3000`);
- A `PGVector` database (aka. postgres with vector extensions);
- A `PGAdmin` webui to manage the db with a GUI;

## Stopping the RAG infra

To stop the basic chat infra, run: `docker compose --profile rag down`


# Running the RAG chat demo

You must have a somewhat modern version of python and Docker installed

- (**If not already present**) Create a venv in your current dir: `python3 -m venv venv`
- (**If not already activated**) Activate the venv: `source venv/bin/activate`
- Install deps: `pip install -r rag_chat/requirements.txt`
- Scrape `docs.docker.com`, chunk the docs, create the embeddings for them, and save them to the `PGVector` DB by running: `python3 rag_chat/main.py`
- Run the demo RAG chat app: `python3 rag_chat/chat.py gemma2:2b`  
  (replace `gemma2:2b` with any model present on [ollama](https://ollama.com/library/))


# Connecting Open WubUI to Ollama

Once logged into the web ui, click on:
- The user icon (bottom left);
- Settings;
- Admin Settings;
- Connections;

and set `http://ollama:11434` as the Ollama API (you can add a OpenAI API key as well if you want)
