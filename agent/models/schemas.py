from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class ReasoningThoughts(BaseModel):
    thoughts: List[str] = Field(
        description="List of reasoning steps about the paper's publishability"
    )
    next_aspect: str = Field(
        description="The next aspect of the paper to analyze"
    )

class PathEvaluation(BaseModel):
    best_path: str = Field(description="ID of the best reasoning path")
    neutral_paths: List[str] = Field(description="IDs of acceptable paths")
    pruned_paths: List[str] = Field(description="IDs of paths to be pruned")
    rationale: str = Field(description="Reasoning for categorization")

class PublishabilityDecision(BaseModel):
    is_publishable: bool
    primary_strengths: List[str]
    critical_weaknesses: List[str]
    recommendation: str

class EvaluationResponse(BaseModel):
    is_publishable: bool
    primary_strengths: List[str]
    critical_weaknesses: List[str]
    recommendation: str
    ai_content_percentage: float
    thought_tree_data: dict 

class ClassifierPrediction(BaseModel):
    conference: str
    rationale: str
    confidence: Optional[float] = None
    thought_process: Optional[List[str]] = None

class ClassificationResponse(BaseModel):
    final_prediction: ClassifierPrediction
    llm_prediction: ClassifierPrediction  
    rag_prediction: ClassifierPrediction
    similarity_prediction: ClassifierPrediction
    metadata: Optional[Dict] = None 