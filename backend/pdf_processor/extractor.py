"""
PDF Text Extractor
Extracts text and metadata from research paper PDFs
Uses LangChain document loaders for better extraction
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

# Try LangChain loaders first (best quality)
try:
    from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Fallback to direct PDF libraries
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class PDFExtractor:
    """Extracts text and metadata from PDF files"""
    
    # Paper metadata mapping (filename -> metadata)
    PAPER_METADATA = {
        "attention_is_all_you_need.pdf": {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
            "year": 2017,
            "venue": "NeurIPS",
            "arxiv_id": "1706.03762"
        },
        "bert.pdf": {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
            "year": 2018,
            "venue": "NAACL",
            "arxiv_id": "1810.04805"
        },
        "llama.pdf": {
            "title": "LLaMA: Open and Efficient Foundation Language Models",
            "authors": ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet", "Marie-Anne Lachaux", "Timothée Lacroix", "Baptiste Rozière", "Naman Goyal", "Eric Hambro", "Faisal Azhar", "Aurelien Rodriguez", "Armand Joulin", "Edouard Grave", "Guillaume Lample"],
            "year": 2023,
            "venue": "arXiv",
            "arxiv_id": "2302.13971"
        },
        "lora.pdf": {
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
            "year": 2021,
            "venue": "ICLR",
            "arxiv_id": "2106.09685"
        },
        "flash_attention.pdf": {
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
            "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Ré"],
            "year": 2022,
            "venue": "NeurIPS",
            "arxiv_id": "2205.14135"
        },
        "vision_transformer.pdf": {
            "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            "authors": ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", "Dirk Weissenborn", "Xiaohua Zhai", "Thomas Unterthiner", "Mostafa Dehghani", "Matthias Minderer", "Georg Heigold", "Sylvain Gelly", "Jakob Uszkoreit", "Neil Houlsby"],
            "year": 2020,
            "venue": "ICLR",
            "arxiv_id": "2010.11929"
        },
        "diffusion_ddpm.pdf": {
            "title": "Denoising Diffusion Probabilistic Models",
            "authors": ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
            "year": 2020,
            "venue": "NeurIPS",
            "arxiv_id": "2006.11239"
        }
    }
    
    def __init__(self):
        """Initialize PDF extractor"""
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file using best available method
        
        Priority:
        1. LangChain PyPDFLoader (best quality, proper spacing)
        2. pdfplumber (good for complex layouts)
        3. pypdf (fallback)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content (cleaned)
        """
        # Method 1: Try LangChain PyPDFLoader (best quality)
        if LANGCHAIN_AVAILABLE:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Combine all pages
                text_content = [doc.page_content for doc in documents]
                full_text = "\n\n".join(text_content)
                
                # LangChain loaders usually have good spacing, but clean up anyway
                cleaned_text = self._clean_extracted_text(full_text)
                return cleaned_text
            except Exception as e:
                print(f"Warning: LangChain loader failed: {e}, trying fallback...")
        
        # Method 2: Try pdfplumber (good for complex layouts)
        if PDFPLUMBER_AVAILABLE:
            try:
                text_content = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        # Extract words individually to preserve spacing
                        words = page.extract_words()
                        if words:
                            text = self._reconstruct_text_from_words(words)
                            if text:
                                text_content.append(text)
                        else:
                            # Fallback to simple text extraction
                            text = page.extract_text()
                            if text:
                                text_content.append(text)
                
                if text_content:
                    full_text = "\n\n".join(text_content)
                    cleaned_text = self._clean_extracted_text(full_text)
                    return cleaned_text
            except Exception as e:
                print(f"Warning: pdfplumber failed: {e}, trying pypdf...")
        
        # Method 3: Fallback to pypdf
        if PYPDF_AVAILABLE:
            try:
                text_content = []
                with open(pdf_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                
                if text_content:
                    full_text = "\n\n".join(text_content)
                    cleaned_text = self._clean_extracted_text(full_text)
                    return cleaned_text
            except Exception as e:
                print(f"Error extracting text with pypdf: {e}")
        
        return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text to fix word spacing issues
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text with proper spacing
        """
        import re
        
        # Step 1: Fix lowercase-uppercase transitions (most common)
        # e.g., "Inthissection" -> "In this section"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Step 2: Fix uppercase-lowercase-uppercase patterns (word boundaries)
        # e.g., "ThisIsATest" -> "This Is A Test"
        text = re.sub(r'([A-Z][a-z]+)([A-Z][a-z])', r'\1 \2', text)
        
        # Step 3: Fix number-letter boundaries
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
        # Step 4: Fix punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Step 5: Fix multiple spaces and newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        
        # Step 6: Clean up lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Step 7: Remove excessive empty lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Step 8: Fix merged words (aggressive splitting using word frequency)
        # Extended list of common English words
        common_words = [
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use',
            'that', 'with', 'have', 'this', 'will', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were',
            'which', 'their', 'there', 'these', 'those', 'where', 'while', 'after', 'before', 'during', 'under', 'until', 'which', 'within', 'without',
            'about', 'above', 'across', 'after', 'again', 'along', 'among', 'around', 'because', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
            'attention', 'mechanism', 'model', 'paper', 'section', 'method', 'result', 'experiment', 'training', 'network', 'neural', 'learning', 'algorithm', 'data', 'performance'
        ]
        
        # Sort by length (longest first) for better matching
        common_words.sort(key=len, reverse=True)
        
        # Recursively split merged words
        def split_merged_word(word, depth=0):
            """Recursively split a word that might contain multiple merged words"""
            if depth > 10:  # Prevent infinite recursion
                return word
            if len(word) < 6:  # Very short words are probably fine
                return word
            
            word_lower = word.lower()
            splits = []
            
            # Find all possible split points (common words within the word)
            for common in common_words:
                if len(common) < 3:  # Skip very short words
                    continue
                # Find all occurrences
                start = 0
                while True:
                    idx = word_lower.find(common, start)
                    if idx == -1:
                        break
                    # Make sure we're not at the very start (allow 1-2 chars)
                    if 1 < idx < len(word) - len(common):
                        splits.append((idx, len(common)))
                    start = idx + 1
            
            if splits:
                # Sort by position and use the first valid split
                splits.sort()
                best_idx, best_len = splits[0]
                
                # Split at this position
                part1 = word[:best_idx]
                part2 = word[best_idx:]
                # Recursively split both parts
                part1_split = split_merged_word(part1, depth+1) if len(part1) > 6 else part1
                part2_split = split_merged_word(part2, depth+1) if len(part2) > 6 else part2
                return part1_split + ' ' + part2_split
            
            return word
        
        # Apply splitting to long lowercase words (15+ chars are almost certainly merged)
        def replace_long_word(match):
            word = match.group(0)
            if len(word) >= 12:  # Split words 12+ chars
                return split_merged_word(word)
            return word
        
        # Split long lowercase sequences
        text = re.sub(r'\b[a-z]{12,}\b', replace_long_word, text)
        
        # Step 9: Final cleanup
        text = re.sub(r' +', ' ', text)  # Clean up any new double spaces
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on new lines
        
        return text.strip()
    
    def _reconstruct_text_from_words(self, words: List[Dict]) -> str:
        """
        Reconstruct text from word list with proper spacing
        
        Args:
            words: List of word dictionaries from pdfplumber
            
        Returns:
            Reconstructed text with proper spacing
        """
        if not words:
            return ""
        
        import re
        lines = []
        current_line_words = []
        current_y = None
        
        for i, word in enumerate(words):
            word_text = word.get('text', '').strip()
            if not word_text:
                continue
            
            word_y = word.get('top', 0)
            word_x0 = word.get('x0', 0)
            word_x1 = word.get('x1', 0)
            
            # If y-coordinate changed significantly, it's a new line
            if current_y is not None and abs(word_y - current_y) > 3:
                if current_line_words:
                    line_text = self._join_words_with_spacing(current_line_words)
                    lines.append(line_text)
                current_line_words = [(word_text, word_x0, word_x1)]
                current_y = word_y
            else:
                # Same line
                if current_line_words:
                    # Check spacing from previous word
                    prev_text, prev_x0, prev_x1 = current_line_words[-1]
                    gap = word_x0 - prev_x1
                    
                    # Determine if we need a space
                    # If gap is very small (< 1), words might be merged
                    # If gap is larger, add space
                    if gap > 1:
                        current_line_words.append((word_text, word_x0, word_x1))
                    else:
                        # Small gap - might be merged, but add space for readability
                        current_line_words.append((word_text, word_x0, word_x1))
                else:
                    current_line_words = [(word_text, word_x0, word_x1)]
                    current_y = word_y
        
        # Add last line
        if current_line_words:
            line_text = self._join_words_with_spacing(current_line_words)
            lines.append(line_text)
        
        text = '\n'.join(lines)
        
        # Additional cleanup for any remaining merged words
        text = self._clean_extracted_text(text)
        
        return text
    
    def _join_words_with_spacing(self, word_list: List[tuple]) -> str:
        """
        Join words with proper spacing based on their positions
        
        Args:
            word_list: List of (text, x0, x1) tuples
            
        Returns:
            Joined text with spaces
        """
        if not word_list:
            return ""
        
        result = []
        for i, (text, x0, x1) in enumerate(word_list):
            if i == 0:
                result.append(text)
            else:
                prev_text, prev_x0, prev_x1 = word_list[i-1]
                gap = x0 - prev_x1
                
                # Add space if gap is reasonable, otherwise might be same word
                if gap > 1:
                    result.append(' ' + text)
                else:
                    # Very small gap - might be merged word
                    # Try to detect if it's actually a separate word
                    # by checking for capital letters or common patterns
                    if text and text[0].isupper() and prev_text and prev_text[-1].islower():
                        result.append(' ' + text)  # "wordWord" -> "word Word"
                    else:
                        result.append(' ' + text)  # Add space anyway for safety
        
        return ''.join(result)
    
    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with metadata
        """
        filename = os.path.basename(pdf_path)
        
        # Get metadata from our mapping
        metadata = self.PAPER_METADATA.get(filename, {})
        
        # Add file info
        metadata.update({
            "filename": filename,
            "file_path": pdf_path,
            "file_size": os.path.getsize(pdf_path),
            "extracted_at": datetime.now().isoformat()
        })
        
        # Try to get metadata from PDF if available
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    if not metadata.get("title") and pdf_meta.get("/Title"):
                        metadata["title"] = pdf_meta.get("/Title")
                    if not metadata.get("authors") and pdf_meta.get("/Author"):
                        metadata["authors"] = [pdf_meta.get("/Author")]
        except:
            pass
        
        return metadata
    
    def process_paper(self, pdf_path: str) -> Dict:
        """
        Process a paper: extract text and metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with text and metadata
        """
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        text = self.extract_text(pdf_path)
        metadata = self.extract_metadata(pdf_path)
        
        return {
            "text": text,
            "metadata": metadata,
            "text_length": len(text),
            "word_count": len(text.split())
        }


if __name__ == "__main__":
    # Test extraction
    extractor = PDFExtractor()
    
    papers_dir = "data/papers"
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            result = extractor.process_paper(pdf_path)
            
            print(f"\n{'='*60}")
            print(f"Paper: {result['metadata'].get('title', filename)}")
            print(f"Authors: {', '.join(result['metadata'].get('authors', []))}")
            print(f"Year: {result['metadata'].get('year', 'Unknown')}")
            print(f"Text length: {result['text_length']:,} characters")
            print(f"Word count: {result['word_count']:,} words")
            print(f"{'='*60}\n")

