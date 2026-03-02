from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from pathlib import Path
import os

class PaperParser:
    def __init__(self):
        # Docling handles the heavy lifting of layout analysis
        self.converter = DocumentConverter()

    def parse_pdf(self, pdf_path: Path):
        print(f"Parsing document: {pdf_path.name}...")
        
        # Convert PDF to structured Markdown
        result = self.converter.convert(pdf_path)
        
        # We export to markdown because it preserves headings and tables 
        # which is better for RAG "chunking" later
        markdown_content = result.document.export_to_markdown()
        
        return markdown_content

if __name__ == "__main__":
    # Test with one of the PDFs you just downloaded
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found in data/pdfs. Run arxiv_client.py first!")
    else:
        parser = PaperParser()
        # Parse the first PDF found
        content = parser.parse_pdf(pdf_files[0])
        
        # Print a preview of the structured text
        print("\n--- Document Preview (First 500 chars) ---")
        print(content[:500])
        print("\n--- End Preview ---")