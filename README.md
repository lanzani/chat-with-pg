# Chat with Paul Graham

## Requirements

1. `poetry install` to install the dependencies
2. Install [Ollama](https://ollama.com/), the model used is llama3.1

## Populate db
1. Set your `OPENAI_API_KEY` env variable
2. Run `python populate_db.py`

To update the db add the new pdf to the "essay" folder and run again `python populate_db.py`.

# Chat with PG
1. Set your `OPENAI_API_KEY` env variable
2. Run `python chat_with_pg.py`

