import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from data_processor import DataProcessor
from rag_handler import RAGHandler
from visualization_agent import VisualizationAgent
import json
import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set page config
st.set_page_config(
    page_title="AI Data Analyst Agent", 
    page_icon="📊", 
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "data_processor" not in st.session_state:
    st.session_state.data_processor = None

if "rag_handler" not in st.session_state:
    st.session_state.rag_handler = None

if "viz_agent" not in st.session_state:
    st.session_state.viz_agent = None

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}

if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False

# App title and description
st.title("🤖 AI Data Analyst Agent")
st.markdown("""
Upload documents (CSV, XLSX, PDF, DOC, TXT) and get intelligent analysis, answers to questions, and visualizations.
The agent uses advanced RAG techniques and works autonomously to provide insights.
""")

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Together.ai API Key
    st.subheader("Together.ai API Setup")
    together_api_key = st.text_input(
        "Enter Together.ai API Key", 
        type="password",
        help="Required for accessing the Llama-4-Maverick model"
    )
    
    if together_api_key:
        os.environ["TOGETHER_API_KEY"] = together_api_key
        st.session_state.api_key_configured = True
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Please enter your Together.ai API key")
    
    st.markdown("---")
    
    # File upload section
    st.header("📁 Upload Documents")
    st.markdown("Supported formats: CSV, XLSX, PDF, DOC, DOCX, TXT")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["csv", "xlsx", "xls", "pdf", "doc", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload multiple files for comprehensive analysis"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} files uploaded")
        
        # Display uploaded files
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # Size in KB
            st.write(f"• {file.name} ({file_size:.1f} KB)")
        
        # Process files button
        if st.button("🔄 Process Documents", use_container_width=True):
            if not st.session_state.api_key_configured:
                st.error("❌ Please configure Together.ai API key first")
            else:
                with st.spinner("🔄 Processing documents... This may take a while."):
                    try:
                        # Initialize components
                        st.session_state.data_processor = DataProcessor()
                        st.session_state.rag_handler = RAGHandler(together_api_key)
                        st.session_state.viz_agent = VisualizationAgent(together_api_key)
                        
                        # Process uploaded files
                        processed_data = st.session_state.data_processor.process_files(uploaded_files)
                        st.session_state.uploaded_data = processed_data
                        
                        # Setup RAG for different data types
                        st.session_state.rag_handler.setup_rag_pipeline(processed_data)
                        
                        st.success("✅ Documents processed successfully! You can now ask questions.")
                        
                        # Show data summary
                        st.subheader("📊 Data Summary")
                        for file_name, data_info in processed_data.items():
                            if data_info['type'] == 'structured':
                                df = data_info['data']
                                st.write(f"**{file_name}**: {df.shape[0]} rows, {df.shape[1]} columns")
                            elif data_info['type'] == 'text':
                                text_len = len(data_info['data'])
                                st.write(f"**{file_name}**: {text_len} characters")
                            elif data_info['type'] == 'image':
                                st.write(f"**{file_name}**: Image processed with OCR")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")

# Main chat interface
st.header("💬 Chat with Your Data")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "visualization" in message:
            # Display text response
            st.markdown(message["content"])
            # Display visualization if present
            if message["visualization"]:
                st.plotly_chart(message["visualization"], use_container_width=True)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask questions about your data, request analysis, or ask for visualizations..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if system is ready
    if not st.session_state.api_key_configured:
        response = "Please configure your Together.ai API key in the sidebar first."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    elif not st.session_state.rag_handler:
        response = "Please upload and process documents first."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(" Analyzing your request...(This might take fewminutes)"):
                try:
                    # Get response from RAG handler
                    response_data = st.session_state.rag_handler.get_response(
                        prompt, 
                        st.session_state.uploaded_data
                    )
                    
                    # Check if visualization is needed
                    viz_needed = False
                    if st.session_state.viz_agent:
                        viz_needed = st.session_state.viz_agent.should_create_visualization(prompt)
                    
                    response_text = response_data.get('answer', 'No response generated.')
                    visualization = None
                    
                    if viz_needed and st.session_state.uploaded_data and st.session_state.viz_agent:
                        # Create visualization
                        try:
                            visualization = st.session_state.viz_agent.create_visualization(
                                prompt, 
                                st.session_state.uploaded_data,
                                response_data.get('context', '')
                            )
                        except Exception as viz_error:
                            st.warning(f"⚠️ Could not create visualization: {str(viz_error)}")
                    
                    # Display response
                    st.markdown(response_text)

                    # Display visualization if created
                    if visualization:
                        st.plotly_chart(visualization, use_container_width=True)
                    
                    # Add to message history
                    message_data = {
                        "role": "assistant", 
                        "content": response_text
                    }
                    if visualization:
                        message_data["visualization"] = visualization
                    
                    st.session_state.messages.append(message_data)
                
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Data preview section
if st.session_state.uploaded_data:
    st.header("📋 Data Preview")
    
    # Create tabs for different data types
    structured_data = {k: v for k, v in st.session_state.uploaded_data.items() 
                      if v.get('type') == 'structured'}
    
    if structured_data:
        tabs = st.tabs(list(structured_data.keys()))
        
        for tab, (file_name, data_info) in zip(tabs, structured_data.items()):
            with tab:
                df = data_info['data']
                st.dataframe(df.head(100), use_container_width=True)
                
                # Basic statistics
                if st.button(f"📊 Show Statistics for {file_name}"):
                    st.subheader("Basic Statistics")
                    st.write(df.describe())
                    
                    # Data types
                    st.subheader("Data Types")
                    st.write(df.dtypes.to_frame('Data Type'))

# Footer
st.markdown("---")
st.markdown("""
### 🔧 System Components
- **LLM**: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free (Together.ai)
- **Embeddings**: FastEmbed (Local)
- **Vector DB**: FAISS
- **Framework**: LangChain
- **Visualizations**: Plotly
""")
