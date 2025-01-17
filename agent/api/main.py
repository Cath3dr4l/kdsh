from dotenv import load_dotenv

load_dotenv()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from services.evaluation_service import EvaluationService
from models.config import ModelConfig, ModelProvider
from models.schemas import EvaluationResponse, ClassificationResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


from services.final_classifier import FinalClassifier
from models.schemas import ClassificationResponse
from services.classification_service import ClassificationService


# Initialize service
classification_service = ClassificationService()


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize evaluation service
reasoning_config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4o-mini")

critic_config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4o")

evaluation_service = EvaluationService(reasoning_config, critic_config)


class TextInput(BaseModel):
    content: str


import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool executor for concurrent processing
thread_pool = ThreadPoolExecutor(max_workers=4)


@app.post("/evaluate/pdf", response_model=EvaluationResponse)
async def evaluate_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run evaluation in the thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool, lambda: asyncio.run(evaluation_service.evaluate_pdf(tmp_path))
        )
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/evaluate/text", response_model=EvaluationResponse)
async def evaluate_text(input_data: TextInput):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        lambda: asyncio.run(evaluation_service.evaluate_text(input_data.content)),
    )
    return result


@app.post("/classify/pdf", response_model=ClassificationResponse)
async def classify_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run classification in the thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool,
            lambda: asyncio.run(classification_service.classify_pdf(tmp_path)),
        )
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/classify/text", response_model=ClassificationResponse)
async def classify_text(input_data: TextInput):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool,
        lambda: asyncio.run(classification_service.classify_text(input_data.content)),
    )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
