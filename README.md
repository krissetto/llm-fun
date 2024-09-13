# Running the infra

Run the compose file to spin up an `ollama` server (exposed on port `11434`) and an `openwebui` instance (exposed on port `3000`): `docker compose up -d`

You should now be able to visit `localhost:3000` and create your own local openwebui account

# Running the chat demo

You must have a somewhat modern version of python installed, and Docker

- create a venv in your current dir: `python3 -m venv venv`
- activate the venv: `source venv/bin/activate`
- install deps: `pip install -r requirements.txt`
- run the demo chat app: `python3 main.py`


# Connecting openwebui to ollama

Once logged into the web ui, click on:
- The user icon (bottom left);
- Admin Panel;
- Connections;

and set `http://ollama:11434` as the Ollama API (you can add a OpenAI API key as well if you want)
