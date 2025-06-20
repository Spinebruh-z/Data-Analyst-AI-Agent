import os
import re
from typing import cast, Union, List, Dict, Any
import faiss
import numpy as np
from langchain.schema import Document , BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_together import ChatTogether
from pydantic import SecretStr
import pandas as pd
import together
import json

class RAGHandler:
    """
    Handles RAG pipeline for different data types using Together.ai and FastEmbed
    """
    
    def __init__(self, together_api_key: str):
        self.together_client = together.Together(api_key=together_api_key)
        self.together_api_key = together_api_key
        self.embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vector_stores = {}
        self.csv_agents = {}
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.llm = self._create_together_llm()
    
    def setup_rag_pipeline(self, processed_data: Dict[str, Any]):
        """Setup RAG pipeline for different data types"""
        try:
            # Process structured data (CSV/Excel)
            self._setup_structured_data_rag(processed_data)
            
            # Process text data (PDF, DOC, TXT)
            self._setup_text_data_rag(processed_data)
            
            # Process image data (OCR text)
            self._setup_image_data_rag(processed_data)
            
            #Debug
            print("Vector store keys:", list(self.vector_stores.keys()))
            for k, store in self.vector_stores.items():
                print(f"Store {k} has {len(store.texts)} documents.")

        except Exception as e:
            raise Exception(f"Error setting up RAG pipeline: {str(e)}")
    
    def _setup_structured_data_rag(self, processed_data: Dict[str, Any]):
        """Setup RAG for structured data using CSV agents"""
        for file_name, data_info in processed_data.items():
            if data_info.get('type') == 'structured':
                try:
                    df = data_info['data']
                    
                    # Create CSV agent for direct querying
                    # Save dataframe to temporary CSV for the agent
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        
                        # Create CSV agent with Together.ai LLM
                        self.csv_agents[file_name] = create_csv_agent(
                            llm=self.llm,
                            path=tmp_file.name,
                            verbose=True,
                            agent_type="zero-shot-react-description",
                            allow_dangerous_code=True
                        )
                    
                    # Also create vector embeddings for the structured data summary
                    self._create_structured_embeddings(file_name, df)
                    
                except Exception as e:
                    print(f"Error setting up structured data RAG for {file_name}: {str(e)}")
    
    def _setup_text_data_rag(self, processed_data: Dict[str, Any]):
        """Setup RAG for text data"""
        text_documents = []
        
        for file_name, data_info in processed_data.items():
            if data_info.get('type') == 'text':
                try:
                    chunks = data_info['metadata']['chunks']
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                'source': file_name,
                                'chunk_id': i,
                                'type': 'text'
                            }
                        )
                        text_documents.append(doc)
                except Exception as e:
                    print(f"Error processing text data for {file_name}: {str(e)}")
        
        if text_documents:
            try:
                # Create vector store for text documents
                self.vector_stores['text'] = self._create_faiss_vectorstore(text_documents)
            except Exception as e:
                print(f"Error creating vector store for text data: {str(e)}")
    
    def _setup_image_data_rag(self, processed_data: Dict[str, Any]):
        """Setup RAG for image OCR text"""
        image_documents = []
        
        for file_name, data_info in processed_data.items():
            if data_info.get('type') == 'image':
                try:
                    chunks = data_info['metadata'].get('chunks', [])
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():  # Only add non-empty chunks
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    'source': file_name,
                                    'chunk_id': i,
                                    'type': 'image_ocr',
                                    'confidence': data_info['metadata'].get('ocr_confidence', 0)
                                }
                            )
                            image_documents.append(doc)
                except Exception as e:
                    print(f"Error processing image data for {file_name}: {str(e)}")
        
        if image_documents:
            try:
                # Create vector store for image OCR text
                self.vector_stores['image'] = self._create_faiss_vectorstore(image_documents)
            except Exception as e:
                print(f"Error creating vector store for image data: {str(e)}")
    
    def _create_structured_embeddings(self, file_name: str, df: pd.DataFrame):
        """Create embeddings for structured data metadata and summary"""
        try:
            # Create summary documents for the dataframe
            summary_docs = []
            
            # Data overview
            overview = f"Dataset: {file_name}\n"
            overview += f"Shape: {df.shape[0]} rows and {df.shape[1]} columns\n"
            overview += f"Columns: {', '.join(df.columns)}\n"
            
            # Column descriptions
            for col in df.columns:
                col_info = f"Column '{col}': "
                col_info += f"Type: {df[col].dtype}, "
                
                if df[col].dtype in ['int64', 'float64']:
                    col_info += f"Range: {df[col].min()} to {df[col].max()}, "
                    col_info += f"Mean: {df[col].mean():.2f}, "
                    col_info += f"Missing values: {df[col].isnull().sum()}"
                else:
                    unique_count = df[col].nunique()
                    col_info += f"Unique values: {unique_count}, "
                    if unique_count <= 10:
                        col_info += f"Values: {', '.join(map(str, df[col].unique()))}, "
                    col_info += f"Missing values: {df[col].isnull().sum()}"
                
                summary_docs.append(Document(
                    page_content=col_info,
                    metadata={'source': file_name, 'type': 'column_info', 'column': col}
                ))
            
            # Add overview
            summary_docs.append(Document(
                page_content=overview,
                metadata={'source': file_name, 'type': 'dataset_overview'}
            ))
            
            # Sample data
            sample_data = f"Sample data from {file_name}:\n"
            sample_data += df.head().to_string()
            summary_docs.append(Document(
                page_content=sample_data,
                metadata={'source': file_name, 'type': 'sample_data'}
            ))
            
            # Create vector store for structured data metadata
            if 'structured' not in self.vector_stores:
                self.vector_stores['structured'] = self._create_faiss_vectorstore(summary_docs)
            else:
                # Add to existing vector store
                new_vectorstore = self._create_faiss_vectorstore(summary_docs)
                self.vector_stores['structured'].merge_from(new_vectorstore)
                
        except Exception as e:
            print(f"Error creating structured embeddings for {file_name}: {str(e)}")
    
    def _create_faiss_vectorstore(self, documents: List[Document]):
        """Create FAISS vector store with FastEmbed embeddings"""
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings using FastEmbed
            embeddings_list = self.embedding_model.embed_documents(texts)
            embeddings_array = np.array(embeddings_list)
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            embeddings_array = np.ascontiguousarray(embeddings_array.astype('float32'))
            # Add vectors to index (with compatibility across FAISS versions)
            try:
                index.add(embeddings_array) # type: ignore
            except TypeError:
                index.add(embeddings_array.shape[0], embeddings_array)
            
            # Create custom FAISS wrapper
            vectorstore = FAISSVectorStore(
                index=index,
                texts=texts,
                metadatas=metadatas,
                embedding_model=self.embedding_model
            )
            
            return vectorstore
            
        except Exception as e:
            raise Exception(f"Error creating FAISS vectorstore: {str(e)}")
    
    def _create_together_llm(self):
        return ChatTogether(
            model=self.model_name,
            temperature=0.3,
            api_key=SecretStr(self.together_api_key)  
        )
    
    def get_response(self, query: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get response from appropriate RAG pipeline based on query"""
        try:
            # Determine query type and route accordingly
            response_data = {
                'answer': '',
                'context': '',
                'sources': [],
                'data_type': 'mixed'
            }
            
            # Check if query is about structured data analysis
            structured_keywords = ['analyze', 'statistics', 'mean', 'average', 'count', 'sum', 
                                 'correlation', 'distribution', 'plot', 'chart', 'graph']
            
            is_structured_query = any(keyword in query.lower() for keyword in structured_keywords)

            print("Vector stores available:", list(self.vector_stores.keys()))

            if is_structured_query and self.csv_agents:
                # Use CSV agent for structured data queries
                response_data.update(self._handle_structured_query(query, processed_data))
            else:
                # Use vector search for general queries
                response_data.update(self._handle_general_query(query))
            
            return response_data
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'context': '',
                'sources': [],
                'data_type': 'error'
            }
    
    def _handle_structured_query(self, query: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries about structured data using CSV agents"""
        try:
            responses = []
            sources = []
            
            # Try each CSV agent
            for file_name, agent in self.csv_agents.items():
                try:
                    agent_response = agent.run(query)
                    responses.append(f"From {file_name}: {agent_response}")
                    sources.append(file_name)
                except Exception as e:
                    print(f"Error with CSV agent for {file_name}: {str(e)}")
            
            if responses:
                # Combine responses from multiple datasets
                combined_response = "\n\n".join(responses)
                
                # Get additional context from vector stores
                context = self._get_vector_context(query)
                
                # Generate final response using Together.ai
                final_response = self._generate_final_response(query, combined_response, context)
                
                return {
                    'answer': final_response,
                    'context': combined_response,
                    'sources': sources,
                    'data_type': 'structured'
                }
            else:
                return {
                    'answer': "I couldn't analyze the structured data. Please try rephrasing your query.",
                    'context': '',
                    'sources': [],
                    'data_type': 'structured'
                }
                
        except Exception as e:
            return {
                'answer': f"Error handling structured query: {str(e)}",
                'context': '',
                'sources': [],
                'data_type': 'error'
            }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries using vector search"""
        try:
            # Get relevant context from all vector stores
            all_contexts = []
            all_sources = []
            
            for store_type, vectorstore in self.vector_stores.items():
                try:
                    results = vectorstore.similarity_search(query, k=3)
                    for result in results:
                        all_contexts.append(result['text'])
                        all_sources.append(result['metadata'].get('source', 'Unknown'))
                except Exception as e:
                    print(f"Error searching {store_type} vectorstore: {str(e)}")
            
            print("Context retrieved:", all_contexts)


            if all_contexts:
                context = "\n\n".join(all_contexts)
                response = self._generate_final_response(query, "", context)
                
                return {
                    'answer': response,
                    'context': context,
                    'sources': list(set(all_sources)),
                    'data_type': 'text'
                }
            else:
                return {
                    'answer': "I couldn't find relevant information in the uploaded documents. Please make sure your documents are processed and try rephrasing your query.",
                    'context': '',
                    'sources': [],
                    'data_type': 'none'
                }
                
        except Exception as e:
            return {
                'answer': f"Error handling general query: {str(e)}",
                'context': '',
                'sources': [],
                'data_type': 'error'
            }
    
    def _get_vector_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from vector stores"""
        contexts = []
        
        for store_type, vectorstore in self.vector_stores.items():
            try:
                results = vectorstore.similarity_search(query, k=2)
                for result in results:
                    contexts.append(result['text'])
            except Exception as e:
                print(f"Error getting context from {store_type}: {str(e)}")
        
        return "\n\n".join(contexts[:max_results])
    
    def _generate_final_response(self, query: str, data_analysis: str, context: str) -> str:
        """Generate final response using Together.ai LLM"""
        try:
            prompt = f"""You are an expert data analyst AI assistant. You are given a user's query along with a summary of analysis performed on structured data.
Always base your answer strictly on the data_analysis provided below. Do not invent or assume any data not shown. Do not make guesses about missing values. If the data is insufficient, say so clearly and professionally.
Use concise, clear paragraph(s) and correct spacing. No fancy formatting. Can Make use of bullet points if needed. Be confident, accurate, and to the point.

User Query: {query}

Data Analysis Results:
{data_analysis}

Additional Context:
{context}

Instructions:
- Provide a clear, accurate answer based on the data
- Be concise but comprehensive
- Start directly with the answer, no Step 1, Step 2, etc.
- Provide bullet points or a paragraph summary as per need of answer
- Be confident, professional, user-facing and speak as if you are chatting
- If a chart/visualization will follow, briefly describe what it shows
- Avoid code or simulation references unless specifically asked
- Include specific numbers and insights when available
- If creating recommendations, base them on the data
- If the data doesn't support a conclusion, say so
- Keep the fontstyle consistent all over the response and use appropriate and appealing structure to represent your answer

Now write the Answer:"""

            response = self.llm.invoke(prompt)
            final_response = response.content if hasattr(response, "content") else str(response)

            return final_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"


class FAISSVectorStore:
    """Custom FAISS vector store implementation"""
    
    def __init__(self, index, texts: List[str], metadatas: List[dict], embedding_model):
        self.index = index
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_model = embedding_model
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.texts):  # Valid index
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(scores[0][i])
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def merge_from(self, other_vectorstore):
        """Merge another vector store into this one"""
        try:
            # Add texts and metadata
            self.texts.extend(other_vectorstore.texts)
            self.metadatas.extend(other_vectorstore.metadatas)
            
            # Merge FAISS indices
            self.index.merge_from(other_vectorstore.index)
            
        except Exception as e:
            print(f"Error merging vector stores: {str(e)}")
