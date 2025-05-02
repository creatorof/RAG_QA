import os
import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


persist_dir = './chroma_db'
doc_path = "./db/Indian_Recipe_Cleaned.csv"
model = "llama3.2:latest"
embedding_model = 'all-minilm'

llm = ChatOllama(model=model)  

@st.cache_resource
def initialize_document_retrieval(document_directory):
    embeddings = OllamaEmbeddings(model=embedding_model)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    

    loader = CSVLoader(
        file_path=document_directory,
        metadata_columns=['dish_name', 'description', 'views', 'rating', 'number_of_votes', 'serves',
                          'dietary_info', 'cook_time', 'prep_time'],
        content_columns=['dish_name', 'spice', 'description', 'ingredients', 'instructions'],
        source_column='dish_name'
    )
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name='indian-healthy-recipe-rag',
        persist_directory=persist_dir
    )
    vectorstore.persist()

    return vectorstore

recipe_vector_db = initialize_document_retrieval(doc_path)

recipe_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful recipe assistant that recommends recipes strictly based on the provided context.
     Do not suggest recipes if the answer does not exist within the given context.

     Always mention the following in your response when relevant:
     - Dish Name
     - Description
     - Prep Time
     - Cook Time
     - Serving Size
     - Dietary Info
     - Spice
     - Ingredients
     - Instructions

     Rules:
     - If no relevant recipes are found in the context, respond with: "I couldn't find information about this."
     - If unsure, respond with: "I couldn't find any dish with the given ingredients. You might want to try different ingredients for another result."
     - Do not make up or assume information not found in the context.

     Context:
     {context}
     """),
    ("human", "{input}")
])

recipe_retriever = recipe_vector_db.as_retriever(search_type="similarity")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieval_chain = (
    {
        "context": recipe_retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | recipe_qa_prompt
    | llm
    | StrOutputParser()
)

st.title("🍲 Indian Healthy Recipe Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

if user_input := st.chat_input("Ask about ingredients, recipes, or cooking tips..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Looking through recipes..."):
        try:
            response = retrieval_chain.invoke(user_input)
        except Exception as e:
            response = f"Error: {e}"

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
