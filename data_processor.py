import pandas as pd
import numpy as np
import tempfile
import os
from typing import Dict, Any, List, Tuple, Sequence, cast
import PyPDF2
import docx
import easyocr
from PIL import Image
import io

class DataProcessor:
    """
    Handles processing of different file types for the data analyst agent.
    Ensures no data loss during processing.
    """
    
    def __init__(self):
        self.ocr_reader: easyocr.Reader | None = None
        self.supported_formats = {
            'structured': ['csv', 'xlsx', 'xls'],
            'text': ['pdf', 'doc', 'docx', 'txt'],
            'image': ['png', 'jpg', 'jpeg']
        }
    
    def _init_ocr(self):
        """Initialize OCR reader lazily"""
        if self.ocr_reader is None:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                raise Exception(f"Failed to initialize OCR reader: {str(e)}")
    
    def process_files(self, uploaded_files) -> Dict[str, Any]:
        """
        Process uploaded files and return structured data
        
        Returns:
            Dict with file_name as key and processed data as value
        """
        processed_data = {}
        
        for file in uploaded_files:
            file_name = file.name
            file_extension = file_name.split('.')[-1].lower()
            
            try:
                if file_extension in self.supported_formats['structured']:
                    processed_data[file_name] = self._process_structured_data(file, file_extension)
                elif file_extension in self.supported_formats['text']:
                    processed_data[file_name] = self._process_text_data(file, file_extension)
                elif file_extension in self.supported_formats['image']:
                    processed_data[file_name] = self._process_image_data(file, file_extension)
                else:
                    print(f"Unsupported file format: {file_extension}")
                    
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                processed_data[file_name] = {
                    'type': 'error',
                    'error': str(e),
                    'data': None
                }
        
        return processed_data
    
    def _process_structured_data(self, file, file_extension: str) -> Dict[str, Any]:
        """Process CSV and Excel files"""
        try:
            if file_extension == 'csv':
                # Try different encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252']
                separators = [',', ';', '\t']
                
                df = None
                for encoding in encodings:
                    for sep in separators:
                        try:
                            file.seek(0)  # Reset file pointer
                            df = pd.read_csv(file, encoding=encoding, sep=sep)
                            if df.shape[1] > 1:  # Valid dataframe
                                break
                        except:
                            continue
                    if df is not None and df.shape[1] > 1:
                        break
                
                if df is None:
                    raise ValueError("Could not parse CSV file with any encoding/separator combination")
            
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle missing values appropriately (don't drop to avoid data loss)
            # Just mark them for user awareness
            missing_info = df.isnull().sum()
            
            # Basic data cleaning without data loss
            # Convert numeric columns
            numeric_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().sum() > len(df) * 0.7:  # 70% are numeric
                        df[col] = numeric_series
                        numeric_columns.append(col)
            
            return {
                'type': 'structured',
                'data': df,
                'metadata': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': missing_info.to_dict(),
                    'numeric_columns': numeric_columns,
                    'sample_data': df.head().to_dict('records')
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing structured data: {str(e)}")
    
    def _process_text_data(self, file, file_extension: str) -> Dict[str, Any]:
        """Process text documents (PDF, DOC, DOCX, TXT)"""
        try:
            text_content = ""
            
            if file_extension == 'pdf':
                text_content = self._extract_pdf_text(file)
            elif file_extension in ['doc', 'docx']:
                text_content = self._extract_docx_text(file)
            elif file_extension == 'txt':
                text_content = self._extract_txt_text(file)
            
            # Clean text without losing information
            text_content = text_content.strip()
            
            # Split into chunks for RAG (preserve original structure)
            chunks = self._chunk_text(text_content)
            
            return {
                'type': 'text',
                'data': text_content,
                'metadata': {
                    'length': len(text_content),
                    'word_count': len(text_content.split()),
                    'chunks': chunks,
                    'chunk_count': len(chunks)
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing text data: {str(e)}")
    
    def _process_image_data(self, file, file_extension: str) -> Dict[str, Any]:
        """Process images with OCR"""
        try:
            self._init_ocr()
            
            # Read image
            image = Image.open(file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            if self.ocr_reader is None:
                raise RuntimeError("OCR reader is not initialized.")
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            results = cast(List[Tuple[List[List[int]], str, float]], self.ocr_reader.readtext(np.array(image)))

            # Extract text
            extracted_text = " ".join([result[1] for result in results])
            
            # Get OCR confidence scores
            confidence_scores = [result[2] for result in results]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # Split into chunks for RAG
            chunks = self._chunk_text(extracted_text) if extracted_text else []
            
            return {
                'type': 'image',
                'data': extracted_text,
                'metadata': {
                    'image_size': image.size,
                    'ocr_confidence': avg_confidence,
                    'text_length': len(extracted_text),
                    'word_count': len(extracted_text.split()) if extracted_text else 0,
                    'chunks': chunks,
                    'chunk_count': len(chunks)
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing image data: {str(e)}")
    
    def _extract_pdf_text(self, file) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
        return text
    
    def _extract_docx_text(self, file) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def _extract_txt_text(self, file) -> str:
        """Extract text from TXT file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    file.seek(0)
                    text = file.read().decode(encoding)
                    return text
                except UnicodeDecodeError:
                    continue
            raise Exception("Could not decode text file with any encoding")
        except Exception as e:
            raise Exception(f"Error extracting TXT text: {str(e)}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks for RAG processing while preserving context
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n']
                for i in range(end, start + chunk_size // 2, -1):
                    if text[i] in sentence_endings:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def get_data_summary(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all processed data"""
        summary = {
            'total_files': len(processed_data),
            'file_types': {},
            'structured_data_info': {},
            'text_data_info': {},
            'image_data_info': {}
        }
        
        for file_name, data_info in processed_data.items():
            data_type = data_info.get('type', 'unknown')
            
            if data_type not in summary['file_types']:
                summary['file_types'][data_type] = 0
            summary['file_types'][data_type] += 1
            
            if data_type == 'structured':
                metadata = data_info.get('metadata', {})
                summary['structured_data_info'][file_name] = {
                    'rows': metadata.get('shape', [0, 0])[0],
                    'columns': metadata.get('shape', [0, 0])[1],
                    'column_names': metadata.get('columns', []),
                    'missing_values': sum(metadata.get('missing_values', {}).values())
                }
            
            elif data_type == 'text':
                metadata = data_info.get('metadata', {})
                summary['text_data_info'][file_name] = {
                    'length': metadata.get('length', 0),
                    'word_count': metadata.get('word_count', 0),
                    'chunks': metadata.get('chunk_count', 0)
                }
            
            elif data_type == 'image':
                metadata = data_info.get('metadata', {})
                summary['image_data_info'][file_name] = {
                    'ocr_confidence': metadata.get('ocr_confidence', 0),
                    'text_extracted': metadata.get('text_length', 0) > 0,
                    'word_count': metadata.get('word_count', 0)
                }
        
        return summary
