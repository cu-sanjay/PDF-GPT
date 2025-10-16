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
from datetime import datetime
import io

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 GOOGLE_API_KEY not found in environment variables")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Error configuring Google AI: {str(e)}")
    st.stop()

@st.cache_resource
def get_embeddings():
    """Get cached embeddings model"""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def get_llm_model():
    """Get cached LLM model"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3
    )

def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = {}

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files with error handling"""
    text = ""
    try:
        if not pdf_docs:
            st.warning("📄 Please upload at least one PDF file")
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
                        st.warning(f"⚠️ Could not extract text from page {page_num + 1} of {pdf.name}")
                        continue
            except Exception as e:
                st.error(f"❌ Error reading PDF {pdf.name}: {str(e)}")
                continue
        
        if not text.strip():
            st.error("❌ No text could be extracted from the uploaded PDFs")
            return ""
            
        return text
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
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
            st.error("❌ Could not create text chunks from the extracted text")
            return []
            
        return chunks
    except Exception as e:
        st.error(f"❌ Error creating text chunks: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Create and save vector store with error handling"""
    try:
        if not text_chunks:
            st.error("❌ No text chunks available to create vector store")
            return False
        
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return False

def get_conversational_chain():
    """Create conversational chain with error handling"""
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

        model = get_llm_model()
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        return model, prompt
    except Exception as e:
        st.error(f"❌ Error creating conversational chain: {str(e)}")
        return None, None

def create_download_link(content, filename, label):
    """Create a download button for content"""
    buffer = io.BytesIO()
    buffer.write(content.encode())
    buffer.seek(0)
    
    st.download_button(
        label=label,
        data=buffer,
        file_name=filename,
        mime="text/plain"
    )

def summarize_pdf():
    """Generate PDF summary"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("📄 Please upload and process PDF files first")
            return
        
        with st.spinner("📝 Generating summary..."):
            embeddings = get_embeddings()
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            docs = new_db.similarity_search("summary main points key information", k=10)
            
            if not docs:
                st.warning("⚠️ No content found for summarization")
                return
            
            model, _ = get_conversational_chain()
            if not model:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            summary_prompt = f"""
            Please provide a comprehensive summary of the following document content. 
            Include the main points, key findings, and important information:

            {context}

            Summary:
            """
            
            response = model.invoke(summary_prompt)
            
            if response and hasattr(response, 'content'):
                st.success("✅ Summary generated!")
                st.markdown("### 📋 Document Summary")
                st.write(response.content)
                st.info("💡 Scroll down to download your summary")
                
                st.session_state.generated_content['summary'] = response.content
            else:
                st.error("❌ Could not generate summary")
                
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")

def generate_questions():
    """Generate questions from PDF content"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("📄 Please upload and process PDF files first")
            return
        
        with st.spinner("❓ Generating questions..."):
            embeddings = get_embeddings()
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            docs = new_db.similarity_search("main topics concepts important information", k=8)
            
            if not docs:
                st.warning("⚠️ No content found for question generation")
                return
            
            model, _ = get_conversational_chain()
            if not model:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            questions_prompt = f"""
            Based on the following document content, generate 8-10 thoughtful questions that would help someone understand the key concepts and important information. 
            Make the questions clear and specific:

            {context}

            Questions:
            """
            
            response = model.invoke(questions_prompt)
            
            if response and hasattr(response, 'content'):
                st.success("✅ Questions generated!")
                st.markdown("### ❓ Practice Questions")
                
                questions = response.content.split('\n')
                questions = [q.strip() for q in questions if q.strip() and ('?' in q or q.strip().endswith('.'))]
                
                for i, question in enumerate(questions[:10], 1):
                    question = question.lstrip('0123456789.- ')
                    st.markdown(f"**{i}.** {question}")
                    
                    if st.button(f"Get Answer", key=f"answer_{i}"):
                        answer_question(question)
                
                st.info("💡 Scroll down to download your questions")
                st.session_state.generated_content['questions'] = response.content
            else:
                st.error("❌ Could not generate questions")
                
    except Exception as e:
        st.error(f"❌ Error generating questions: {str(e)}")

def generate_mcqs():
    """Generate multiple choice questions"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("📄 Please upload and process PDF files first")
            return
        
        with st.spinner("📝 Generating MCQs..."):
            embeddings = get_embeddings()
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            docs = new_db.similarity_search("key concepts important facts definitions", k=6)
            
            if not docs:
                st.warning("⚠️ No content found for MCQ generation")
                return
            
            model, _ = get_conversational_chain()
            if not model:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            mcq_prompt = f"""
            Based on the following document content, create 5 multiple choice questions with 4 options each (A, B, C, D). 
            Include the correct answer at the end. Format each question clearly:

            {context}

            MCQs:
            """
            
            response = model.invoke(mcq_prompt)
            
            if response and hasattr(response, 'content'):
                st.success("✅ MCQs generated!")
                st.markdown("### 📝 Multiple Choice Questions")
                st.write(response.content)
                st.info("💡 Scroll down to download your MCQs")
                
                st.session_state.generated_content['mcqs'] = response.content
            else:
                st.error("❌ Could not generate MCQs")
                
    except Exception as e:
        st.error(f"❌ Error generating MCQs: {str(e)}")

def generate_notes():
    """Generate short notes from PDF"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("📄 Please upload and process PDF files first")
            return
        
        with st.spinner("📚 Generating notes..."):
            embeddings = get_embeddings()
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            docs = new_db.similarity_search("main concepts key points important information", k=8)
            
            if not docs:
                st.warning("⚠️ No content found for notes generation")
                return
            
            model, _ = get_conversational_chain()
            if not model:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            notes_prompt = f"""
            Create concise, well-organized study notes from the following content. 
            Use bullet points, headings, and clear structure. Focus on key concepts and important information:

            {context}

            Study Notes:
            """
            
            response = model.invoke(notes_prompt)
            
            if response and hasattr(response, 'content'):
                st.success("✅ Notes generated!")
                st.markdown("### 📚 Study Notes")
                st.write(response.content)
                st.info("💡 Scroll down to download your notes")
                
                st.session_state.generated_content['notes'] = response.content
            else:
                st.error("❌ Could not generate notes")
                
    except Exception as e:
        st.error(f"❌ Error generating notes: {str(e)}")

def answer_question(question):
    """Answer a specific question"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("📄 Please upload and process PDF files first")
            return
        
        with st.spinner("🔍 Finding answer..."):
            embeddings = get_embeddings()
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(question)
            
            if not docs:
                st.warning("⚠️ No relevant information found")
                return
            
            model, prompt = get_conversational_chain()
            if not model or not prompt:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = prompt.format(context=context, question=question)
            
            response = model.invoke(formatted_prompt)
            
            if response and hasattr(response, 'content'):
                st.success("✅ Answer found!")
                st.write("**Answer:**")
                st.write(response.content)
            else:
                st.error("❌ Could not generate answer")
                
    except Exception as e:
        st.error(f"❌ Error answering question: {str(e)}")

def user_input(user_question):
    """Process user question and generate response with error handling"""
    try:
        if not user_question.strip():
            st.warning("⚠️ Please enter a question")
            return
        
        if not os.path.exists("faiss_index"):
            st.error("📄 Please upload and process PDF files first")
            return
        
        with st.spinner("🔍 Searching for answer..."):
            embeddings = get_embeddings()
            
            try:
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error("❌ Error loading vector store. Please reprocess your PDFs.")
                return
            
            docs = new_db.similarity_search(user_question)
            
            if not docs:
                st.warning("⚠️ No relevant information found in the uploaded documents")
                return
            
            model, prompt = get_conversational_chain()
            if not model or not prompt:
                return
            
            context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = prompt.format(context=context, question=user_question)
            
            try:
                response = model.invoke(formatted_prompt)
                
                if response and hasattr(response, 'content'):
                    st.success("✅ Answer found!")
                    st.write("**Reply:**")
                    st.write(response.content)
                    
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': response.content,
                        'timestamp': datetime.now()
                    })
                else:
                    st.error("❌ Could not generate a response")
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="PDF-GPT | Chat With Your PDFs",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E3440;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #4C566A;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    .tagline {
        text-align: center;
        color: #5E81AC;
        font-style: italic;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .sidebar-header {
        color: #2E3440;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #5E81AC;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #81A1C1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #ECEFF4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #5E81AC;
    }
    .chat-message {
        background-color: #E5E9F0;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #88C0D0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">PDF-GPT v2.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat With Your PDFs Using AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Perfect for Students • Researchers • Professionals</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="info-box">📚 <b>Study Smarter</b><br>Generate summaries and notes</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-box">🔍 <b>Research Faster</b><br>Ask questions instantly</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-box">📝 <b>Practice Better</b><br>Generate questions & MCQs</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="info-box">⚡ <b>Save Time</b><br>Extract key information quickly</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 AI-Powered Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Summarize PDF", use_container_width=True):
            summarize_pdf()
    
    with col2:
        if st.button("❓ Generate Questions", use_container_width=True):
            generate_questions()
    
    with col3:
        if st.button("📝 Create MCQs", use_container_width=True):
            generate_mcqs()
    
    with col4:
        if st.button("📚 Generate Notes", use_container_width=True):
            generate_notes()
    
    st.markdown("---")
    
    st.markdown("### 💬 Ask Questions About Your PDFs")
    user_question = st.text_input(
        "Enter your question:",
        placeholder="What are the main findings in this research paper?",
        help="Ask any question about your uploaded documents"
    )
    
    if user_question:
        user_input(user_question)
    
    if st.session_state.chat_history:
        with st.expander("📜 Chat History", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                st.markdown(f'<div class="chat-message"><b>Q:</b> {chat["question"]}<br><b>A:</b> {chat["answer"][:200]}...</div>', unsafe_allow_html=True)
    
    if st.session_state.generated_content:
        st.markdown("---")
        st.markdown("### 📥 Download Generated Content")
        
        cols = st.columns(4)
        if 'summary' in st.session_state.generated_content:
            with cols[0]:
                create_download_link(
                    st.session_state.generated_content['summary'],
                    f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "📋 Download Summary"
                )
        
        if 'questions' in st.session_state.generated_content:
            with cols[1]:
                create_download_link(
                    st.session_state.generated_content['questions'],
                    f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "❓ Download Questions"
                )
        
        if 'mcqs' in st.session_state.generated_content:
            with cols[2]:
                create_download_link(
                    st.session_state.generated_content['mcqs'],
                    f"mcqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "📝 Download MCQs"
                )
        
        if 'notes' in st.session_state.generated_content:
            with cols[3]:
                create_download_link(
                    st.session_state.generated_content['notes'],
                    f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "📚 Download Notes"
                )
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">📁 Document Upload</h2>', unsafe_allow_html=True)
        
        st.markdown("**Quick Start Guide:**")
        st.markdown("1. 📤 Upload your PDF files")
        st.markdown("2. ⚙️ Click 'Process Documents'")
        st.markdown("3. 🚀 Use AI features or ask questions")
        
        st.markdown("---")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to analyze"
        )
        
        if pdf_docs:
            st.success(f"✅ Uploaded {len(pdf_docs)} file(s)")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")
        
        if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF file")
            else:
                with st.spinner("⚙️ Processing documents..."):
                    progress_bar = st.progress(0)
                    
                    progress_bar.progress(25)
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text:
                        progress_bar.progress(50)
                        text_chunks = get_text_chunks(raw_text)
                        
                        if text_chunks:
                            progress_bar.progress(75)
                            success = get_vector_store(text_chunks)
                            
                            progress_bar.progress(100)
                            
                            if success:
                                st.success("✅ Documents processed successfully!")
                                st.balloons()
                            else:
                                st.error("❌ Failed to process documents")
                        else:
                            st.error("❌ Failed to create text chunks")
                    else:
                        st.error("❌ Failed to extract text from PDFs")
                    
                    progress_bar.empty()
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About PDF-GPT v2.0")
        st.markdown("**Created by:** Sanjay")
        st.markdown("**Version:** 2.0")
        st.markdown("**Description:** An AI-powered tool that helps you chat with your PDF documents, generate summaries, create study materials, and extract key information instantly.")
        
        st.markdown("**🔗 Links:**")
        st.markdown("• [📂 GitHub Repository](https://github.com/cu-sanjay/PDF-GPT)")
        st.markdown("• [🐛 Report Issues](https://github.com/cu-sanjay/PDF-GPT/issues)")
        
        st.markdown("---")
        
        st.markdown("**🆕 What's New in v2.0:**")
        st.markdown("• Professional UI design")
        st.markdown("• Performance improvements")
        st.markdown("• Export functionality")
        st.markdown("• Chat history")
        st.markdown("• Better error handling")
        
        st.markdown("---")
        
        st.markdown("**🤖 AI Model:** Google Gemini 2.0 Flash")
        st.markdown("**🔒 Privacy:** Your documents are processed securely and locally")
        st.markdown("**💡 Tip:** Upload research papers, textbooks, or any PDF documents to get started!")

if __name__ == "__main__":
    main()
