import streamlit as st
import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks


def get_vector_store(text_chunks,google_api_key):
    genai.configure(api_key=google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    if vector_store:
         print("Completed")
         vector_store.save_local("faiss_index")
    else:
         print("Embeddings not created")
    


def get_conversational_chain(google_api_key):
    genai.configure(api_key=google_api_key)
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,  don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7,google_api_key=google_api_key)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question,google_api_key):
    genai.configure(api_key=google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=google_api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(google_api_key)

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    return response

def main():
        st.set_page_config(page_title="TalkToPDF",page_icon="📄")
        with st.sidebar:
              st.markdown("<h1 style='text-align: center; color:grey;'>Getting Started 🚀</h1>", unsafe_allow_html=True)
              st.markdown("""<span ><font size=2>1. Get Started: Begin by adding your Gemini API key.</font></span>""",unsafe_allow_html=True)
              st.markdown("""<span ><font size=2>2.Explore Documents: Upload a document and fire away with your questions about the content</font></span>""",unsafe_allow_html=True)
              google_api_key = st.text_input("Google API Key", key="chatbot_api_key", type="password")
              "[Get an Google API key](https://makersuite.google.com/app/apikey)"
              genai.configure(api_key=google_api_key)
              uploaded_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
              if st.button("Submit & Process"):
                   with st.spinner("Processing..."):
                         raw_text = get_pdf_text(uploaded_files)
                         #print(raw_text)
                         text_chunks = get_text_chunks(raw_text)
                         get_vector_store(text_chunks,google_api_key)
                         st.success("Done")
              if st.button("Clear Chat History"):
                st.session_state.messages.clear()
              st.markdown("<h1 color:black;'>Lets Connect! 🤝</h1>", unsafe_allow_html=True)
              "[Linkedin](https://www.linkedin.com/in/muvva-thriveni/)" "  \t\t\t\t"  "[GitHub](https://github.com/MuvvaThriveni)"
    
        home_title="Chat with your PDF 🦜📄"
        st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=4>Beta</font></span>""",unsafe_allow_html=True)
        st.caption(" A streamlit chatbot powered by Gemini to talk with PDF 🤖")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input():
                if not google_api_key:
                        st.info("Please add your Google API key to continue.")
                        st.stop()
                response=user_input(prompt,google_api_key)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                msg=response['output_text']
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                     
if __name__ == "__main__":
    main()
