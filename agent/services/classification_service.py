import os
from typing import Optional
import PyPDF2
from models.schemas import ClassificationResponse
from services.final_classifier import FinalClassifier


class ClassificationService:
    def __init__(self):
        self.classifier = FinalClassifier()

    async def classify_pdf(self, pdf_path: str) -> ClassificationResponse:
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
            return await self.classify_text(content)
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    async def classify_text(self, content: str) -> ClassificationResponse:
        try:
            return await self.classifier.classify_with_details(content)
        except Exception as e:
            raise Exception(f"Error classifying text: {str(e)}")
