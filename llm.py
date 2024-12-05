from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
import numpy as np
import faiss
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain_groq import ChatGroq
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd
import pickle # For saving the embeddings
from langchain_core.messages import AIMessage,HumanMessage

st.title("Find the best teacher for your interest...")
load_dotenv()

embedding_file = 'embeddings.pkl'
metadata_file = 'metadata.pkl'

dataset = pd.read_csv("Professors_extracted_data.csv")


# st.title("Hello")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
def generate_text_local_llama(content,query):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature = 0.5)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    # print(reranked_docs)
    # print("Content in fenerate_text is::",content)
    response=chain.invoke({"input_documents":content, "question": query})
    # print(response)
    return response

# Load embeddings and data from your CSV file
def embed(query=None):
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cuda'})






    
    
    if query:
        embed_query = embedding.embed_query(query)
        return np.array(embed_query).astype('float32').reshape(1, -1)  # Reshape here
    
    if os.path.exists(embedding_file) and os.path.exists(metadata_file):
        print("Loading existing embeddings and metadatafile...")
        with open(embedding_file,'rb') as embed_file, open(metadata_file,'rb') as meta_file:
            embedding = pickle.load(embed_file)
            data = pickle.load(meta_file)
    else:
        print("No new embeddings, creating new ones...")
        loader = CSVLoader(file_path='Professors_extracted_data.csv', encoding='utf-8', csv_args={'delimiter': ','})
        data = loader.load()
    # Extract text and compute embeddings
        text = [doc.page_content for doc in data]
        embed_docs = embedding.embed_documents(text)
        embedding = np.array(embed_docs).astype('float32') # No need to reshape for documents.
        print("Model loaded and embeddings done!")
        # Set up FAISS index
        with open(embedding_file,'wb') as embed_file, open(metadata_file,'wb') as meta_file:
            pickle.dump(embedding,embed_file)
            pickle.dump(data,meta_file)
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension) # uses Euclidean distance (L2) for search between neighbour embeddings
    index.reset()  # Clear any existing embeddings
    index.add(embedding)

    return index, data
index,data = embed()

def search_db(query, top_k=100):
    query_embedding = embed(query)
    if query_embedding.size == 0 or query_embedding.shape[1] != index.d:
        print("Query embedding is empty or dimension mismatch.")
        return []
    
    distances, indices = index.search(query_embedding, top_k)
    if indices.size == 0 or indices[0][0] == -1:
        print("No results found.")
        return []
    
    results = [(data[i], distances[0][j]) for j, i in enumerate(indices[0])]
    # print(results)
    return results

# Sample usage with your local Llama for generation and FAISS for retrieval
# query = "I am interested in the field of Electric Vehicles. Which teacher is best for me?"

def template(query):
    template = '''
    - You are a helpful bot whose task is to return the Name,Email,Cabin No. of the teacher who will be best suited to guide the student based on the interest expressed in query. 
    - Give exact match with the research interest.
    - Do not give things related and all. I want exact match.
    - The query is :{query} '''
    return template.format(query=query)
# query = template(query)


# print(final_doc)    

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage("Hello, how can I help you?")
    ]
query = st.chat_input("Enter your interest...")
if query:
    st.session_state.chat_history.append(HumanMessage(content=query))
    input = template(query)
    search_results = search_db(input)
    # Generate text using local Llama
    final_doc=[]
    for content,_ in search_results:
        final_doc.append(content)
    final_response = generate_text_local_llama(final_doc,query)
    response = final_response.get('output_text','No Response')
    st.session_state.chat_history.append(AIMessage(content=response))

for messages in st.session_state.chat_history:
    if isinstance(messages,AIMessage):
        with st.chat_message("AI"):
            st.write(messages.content)
    if isinstance(messages,HumanMessage):
        with st.chat_message("Human"):
            st.write(messages.content)