import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter

"""
1- We need a function to Create a vector DB from our dataset --- DONE
2- We need function that reads/load the vector DB ---
3- We need a function that gets a movie
"""


class movie_rag:
    def __init__(self, df = None, device = None, main_column = "description", chroma_data_path = "Chroma_DB"):
        self.df = df
        self.main_column = main_column
        self.chroma_data_path = chroma_data_path
        
        if device:
            self.device = device 
        else:
            self.device = "cpu"
            
        self.embedding = HuggingFaceBgeEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2",
                                                  model_kwargs = {'device': self.device})
        
    def create_vector_database(self):
        loader = DataFrameLoader(self.df, page_content_column= self.main_column )
        descriptions = loader.load()

        reviews_vector_db = Chroma.from_documents(
            descriptions, self.embedding, persist_directory=self.chroma_data_path
        )
        return reviews_vector_db
    
    def load_vector_database(self):
        reviews_vector_db = Chroma(persist_directory=self.chroma_data_path, embedding_function=self.embedding)
        return reviews_vector_db
    
    def get_movie(self, question, k):
        vector_db = self.load_vector_database()
        relevant_docs = vector_db.similarity_search(question, k=k)
        return relevant_docs