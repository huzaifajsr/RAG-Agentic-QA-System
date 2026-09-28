"""Document loading functionality."""

import os
import logging
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents from file paths."""

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from a PDF file page by page.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of dictionaries containing content and metadata.
        """
        documents = []
        try:
            filename = os.path.basename(file_path)
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "content": text.strip(),
                        "metadata": {
                            "source": filename,
                            "page": i + 1
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise
        return documents

    def load_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Read a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List with a single dictionary containing content and metadata.
        """
        try:
            filename = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text and text.strip():
                    return [{
                        "content": text.strip(),
                        "metadata": {
                            "source": filename
                        }
                    }]
                return []
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            raise

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a document based on its extension.
        
        Args:
            file_path: Path to the document.
            
        Returns:
            List of dictionaries containing content and metadata.
            
        Raises:
            ValueError: If the file type is unsupported.
        """
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext == '.txt':
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
