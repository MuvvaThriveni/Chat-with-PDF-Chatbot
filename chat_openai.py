import streamlit as st
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate

st.set_page_config(page_title="TalkToPDF",page_icon="📄")
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color:black;'>Getting Started 🚀</h1>", unsafe_allow_html=True)
    st.markdown("""<span ><font size=2>1. Get Started: Begin by adding your Gemini API key.</font></span>""",unsafe_allow_html=True)
    st.markdown("""<span ><font size=2>2.Explore Documents: Upload a document and fire away with your questions about the content</font></span>""",unsafe_allow_html=True)
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    uploaded_file = st.file_uploader("Choose a PDF file 📄", accept_multiple_files=False,type="pdf")
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        
    st.divider()
    st.markdown("<h1 color:black;'>Lets Connect! 🤝</h1>", unsafe_allow_html=True)
    "[Linkedin](https://www.linkedin.com/in/muvva-thriveni/)" "  \t\t\t\t"  "[GitHub](https://github.com/MuvvaThriveni)"
    
os.environ['OPENAI_API_KEY']=openai_api_key
openai.api_key = os.environ['OPENAI_API_KEY']

home_title="Chat with your PDF 🦜📄"
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=4>Beta</font></span>""",unsafe_allow_html=True)
st.caption(" A streamlit chatbot powered by GPT LLM to talk with PDF 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if uploaded_file is not None:
        if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your 🔑 OpenAI API key to continue.")
                st.stop()
            with open("temp_pdf_file.pdf", "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            loader=PyPDFLoader(file_path="temp_pdf_file.pdf")
            docs=loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
            splits = text_splitter.split_documents(docs)
            embedding = OpenAIEmbeddings()
            vectordb = FAISS.from_documents(documents=splits,embedding=embedding)
            retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3, "include_metadata": True})
            memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
            qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5),chain_type='stuff',retriever=retriever,memory=memory)
            sys_prompt = "Act as a friendly and helpful Question Answer System. Answer questions about Document they uploaded"
            qa.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(sys_prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response=qa({"question": prompt})
            msg=response['answer']
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
else:
    
    if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your 🔑 OpenAI API key to continue.")
                st.stop()
            else:
                st.info("Please Upload your document 📄 to continue.")
                st.stop()                
