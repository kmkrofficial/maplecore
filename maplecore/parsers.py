"""
MAPLE Document Parsers
======================
Utility to extract plain text from various rich document formats.
Relying on optional dependencies (PyMuPDF, python-docx, beautifulsoup4).
"""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """
    Extract clean plain text from various file formats.
    Supported extensions: .pdf, .docx, .html, .htm, .txt
    
    Args:
        file_path: Path to the document
        
    Returns:
        Extracted text string
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF is required to parse .pdf files. Install with 'pip install maplecore[parsers]' or 'pip install PyMuPDF'.")
            raise
            
        try:
            doc = fitz.open(path)
            pages_text = []
            for page in doc:
                text = page.get_text()
                if text:
                    pages_text.append(text)
            doc.close()
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise

    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            logger.error("python-docx is required to parse .docx files. Install with 'pip install maplecore[parsers]' or 'pip install python-docx'.")
            raise
            
        try:
            doc = docx.Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Failed to parse DOCX {path}: {e}")
            raise
            
    elif ext in [".html", ".htm"]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 is required to parse HTML files. Install with 'pip install maplecore[parsers]' or 'pip install beautifulsoup4'.")
            raise
            
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator="\n\n", strip=True)
        except Exception as e:
            logger.error(f"Failed to parse HTML {path}: {e}")
            raise
            
    else:
        # Fallback to plain text read
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file as plain text {path}: {e}")
            raise
