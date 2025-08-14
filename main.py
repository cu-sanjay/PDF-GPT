import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure Google AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in environment variables")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google AI: {str(e)}")
    st.stop()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files with error handling"""
    text = ""
    try:
        if not pdf_docs:
            st.warning("Please upload at least one PDF file")
            return ""
        
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1} of {pdf.name}")
                        continue
            except Exception as e:
                st.error(f"Error reading PDF {pdf.name}: {str(e)}")
                continue
        
        if not text.strip():
            st.error("No text could be extracted from the uploaded PDFs")
            return ""
            
        return text
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return ""

def get_text_chunks(text):
    """Split text into chunks with error handling"""
    try:
        if not text.strip():
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, 
            chunk_overlap=1000
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            st.error("Could not create text chunks from the extracted text")
            return []
            
        return chunks
    except Exception as e:
        st.error(f"Error creating text chunks: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Create and save vector store with error handling"""
    try:
        if not text_chunks:
            st.error("No text chunks available to create vector store")
            return False
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain():
    """Create conversational chain with error handling using modern approach"""
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not in the provided context, just say "Answer is not available in the context".
        Do not provide wrong answers.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        return model, prompt
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None, None

def user_input(user_question):
    """Process user question and generate response with error handling"""
    try:
        if not user_question.strip():
            st.warning("Please enter a question")
            return
        
        # Check if vector store exists
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process PDF files first")
            return
        
        with st.spinner("Searching for answer..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            try:
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error("Error loading vector store. Please reprocess your PDFs.")
                return
            
            docs = new_db.similarity_search(user_question)
            
            if not docs:
                st.warning("No relevant information found in the uploaded documents")
                return
            
            model, prompt = get_conversational_chain()
            if not model or not prompt:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = prompt.format(context=context, question=user_question)
            
            try:
                response = model.invoke(formatted_prompt)
                
                if response and hasattr(response, 'content'):
                    st.success("Answer found!")
                    st.write("**Reply:**")
                    st.write(response.content)
                else:
                    st.error("Could not generate a response")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">PDF Chat Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your PDF documents using AI</p>', unsafe_allow_html=True)
    
    # Question input
    st.markdown("### Ask a Question")
    user_question = st.text_input(
        "Enter your question about the uploaded PDF files:",
        placeholder="What is the main topic discussed in the document?",
        help="Type your question and press Enter"
    )
    
    if user_question:
        user_input(user_question)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">Document Upload</h2>', unsafe_allow_html=True)
        
        st.markdown("**Instructions:**")
        st.markdown("1. Upload one or more PDF files")
        st.markdown("2. Click 'Process Documents'")
        st.markdown("3. Ask questions about your documents")
        
        st.markdown("---")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload PDF files to analyze"
        )
        
        if pdf_docs:
            st.success(f"Uploaded {len(pdf_docs)} file(s)")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            if not pdf_docs:
                st.error("Please upload at least one PDF file")
            else:
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    
                    # Extract text
                    progress_bar.progress(25)
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text:
                        # Create chunks
                        progress_bar.progress(50)
                        text_chunks = get_text_chunks(raw_text)
                        
                        if text_chunks:
                            # Create vector store
                            progress_bar.progress(75)
                            success = get_vector_store(text_chunks)
                            
                            progress_bar.progress(100)
                            
                            if success:
                                st.success("Documents processed successfully!")
                                st.balloons()
                            else:
                                st.error("Failed to process documents")
                        else:
                            st.error("Failed to create text chunks")
                    else:
                        st.error("Failed to extract text from PDFs")
                    
                    progress_bar.empty()
        
        # Information section
        st.markdown("---")
        st.markdown("**About this app:**")
        st.markdown("This application uses Google's Gemini AI to answer questions about your PDF documents.")
        st.markdown("Your documents are processed locally and securely.")

if __name__ == "__main__":
    main()
