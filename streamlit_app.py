import streamlit as st
from langchain_openai import OpenAIEmbeddings
from streamlit_chromadb_connection import ChromadbConnection
import pandas as pd
import json
import os
import sys

st.title("RAG Sample")

class RAGEstimate:
    def __init__(self, df, mode):
        self.rules = df.rule
        self.mode = mode
        self.category = df.category
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )

    def rules_to_json(self):
        self.database_to_json(self.rules_to_database())

    def rules_to_database(self):
        configuration = {
            "client": "PersistentClient",
            "path": "./.data"
        }
        
        database = st.connection(
            "chromadb",
            type="ChromaDBConnection",
            **configuration,
            embedding_function=self.embeddings
        )
        database.add_texts(self.rules)
        return database

    def database_to_json(self, database):
        data = database.get()
        with open("rules_vector.json", "w") as f:
            json.dump(data, f, indent=4)

    def prompt(self, database):
        docs = database.similarity_search(self.query_to_prompt())
        documents_string = "\n".join([f"-----------------------\n{doc.page_content}" for doc in docs])
        st.write(documents_string)
        return documents_string

    def run(self):
        self.prompt(self.rules_to_database())

    def query_to_prompt(self):
        if self.mode == "litt":
            query = st.text_input("表現を入力してください：")
        elif len(sys.argv) > 1:
            query = sys.argv[1]
        else:
            query = input("表現を入力してください：")
        return query

def main():
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        mode = "litt"
    else:
        df = pd.read_csv("rule1.csv")
        mode = "default"
    rag = RAGEstimate(df, mode)
    rag.run()

if __name__ == "__main__":
    main()
