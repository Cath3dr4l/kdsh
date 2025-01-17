import os
from unstructured.partition.pdf import partition_pdf

def extract_pdf_content(pdf_path: str) -> str:
    """Extract content from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            pdf_infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=2000,
        )
        
        content = ""
        for element in elements:
            content += element.text
            
        if not content.strip():
            raise ValueError("No content extracted from PDF")
            
        return content
        
    except Exception as e:
        raise Exception(f"Error extracting PDF content: {str(e)}") 