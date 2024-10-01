import streamlit as st
import pandas as pd
from movie import movie_rag


movie_obj = movie_rag(device = 'mps')

def main():
    st.title("Find a Movie")
    
    k = st.number_input("Enter the number of Movies", min_value = 1, value=5, step = 1)
    movie_description = st.text_input("Write a movie Description", value = '')
    
    if st.button("Generate Closest Movies"):
        docs = movie_obj.get_movie(movie_description, k=k)
        for movie in docs:
            st.success(f"Movie Name is: {movie.metadata['title']}", icon = "🎬")
            st.success(f"Movie Description is: {movie.page_content}", icon = "📜")
            
if __name__ == '__main__':
    main()
    