from pprint import pprint

from langchain_chroma import Chroma

from populate_db import CHROMA_PATH
from utils import get_embedding_function, MODEL
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

PROMPT_TEMPLATE = """
You are an assistant that provides answers to questions based on
a given context. 

Answer the question based on the context. If you can't answer the
question, reply "I don't know".

Be as concise as possible and go straight to the point.

Context: {context}

Question: {question}
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatOllama(model=MODEL, temperature=0)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    return response_text, sources


def main():

    print("Chat with Paul Graham :)")
    query = ""
    while query != "exit":
        query = input("\nEnter your query (type 'exit' to quit): ")
        if query == "exit":
            break
        response_text, sources = query_rag(query)
        print("\nResponse:")
        print(response_text.content)
        print(f"Sources: {[s for s in sources]}")


if __name__ == "__main__":
    main()
