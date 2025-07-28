import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import wikipedia
import os

# --- CONFIG ---
# You need to set your OpenAI API key as an environment variable or replace below
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')  # Replace with your key or set env var

# --- UI ---
st.title('Context-Aware Wikipedia Chatbot')

# Sidebar for Wikipedia search
topic = st.sidebar.text_input('Wikipedia Topic', 'Artificial intelligence')
if st.sidebar.button('Import Article'):
    try:
        article = wikipedia.page(topic)
        st.session_state['wiki_content'] = article.content
        st.success(f"Imported: {article.title}")
    except Exception as e:
        st.error(f"Error: {e}")

# Show imported content (optional)
if 'wiki_content' in st.session_state:
    with st.expander('Show Imported Wikipedia Content'):
        st.write(st.session_state['wiki_content'][:2000] + '...')

# --- VECTOR STORE SETUP ---
if 'wiki_content' in st.session_state:
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_texts([st.session_state['wiki_content']], embeddings)

    # --- MEMORY ---
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- LLM ---
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    # --- QA CHAIN ---
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    # --- CHAT UI ---
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = st.text_input('Ask a question about the article:')
    if st.button('Send') and user_input:
        result = qa({"question": user_input, "chat_history": st.session_state['chat_history']})
        st.session_state['chat_history'].append((user_input, result['answer']))
        st.write(f"**Bot:** {result['answer']}")

    # Show chat history
    if st.session_state['chat_history']:
        with st.expander('Chat History'):
            for i, (q, a) in enumerate(st.session_state['chat_history']):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
else:
    st.info('Import a Wikipedia article to start chatting!') 