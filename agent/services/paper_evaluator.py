from dotenv import load_dotenv

load_dotenv()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from typing import List, Dict, Any, Optional, Set
import csv
import time
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import textwrap
import asyncio

import networkx as nx
import matplotlib.pyplot as plt
import uuid

from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from utils.ai_content_detector_tool import TextDetectionTool as checker

from models.schemas import PublishabilityDecision, PathEvaluation, ReasoningThoughts
from services.tree_of_thoughts import TreeOfThoughts
from services.pdf_service import extract_pdf_content

from models.config import ModelConfig, ModelProvider
from services.tree_of_thoughts import ThoughtNode


class LLMFactory:
    @staticmethod
    def create_llm(config: ModelConfig) -> BaseChatModel:
        base_llm = None
        if config.provider == ModelProvider.GROQ:
            base_llm = ChatGroq(
                model=config.model_name,
                temperature=config.temperature,
                max_retries=config.max_retries,
            )
        elif config.provider == ModelProvider.GOOGLE:
            base_llm = ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                max_retries=config.max_retries,
            )
        elif config.provider == ModelProvider.OLLAMA:
            base_llm = ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
            )
        elif config.provider == ModelProvider.OPENAI:
            base_llm = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_retries=config.max_retries,
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        return base_llm


class PaperEvaluator:
    def __init__(self, reasoning_config: ModelConfig, critic_config: ModelConfig):
        reasoning_llm = LLMFactory.create_llm(reasoning_config)
        critic_llm = LLMFactory.create_llm(critic_config)

        self.tot = TreeOfThoughts(reasoning_llm, critic_llm)
        self.decision_llm = critic_llm.with_structured_output(PublishabilityDecision)

    async def evaluate_paper(
        self, content: str, regenerate: bool = False
    ) -> PublishabilityDecision:
        root_node = ThoughtNode(content="root", aspect="initial")
        paths_by_depth: Dict[int, List[List[ThoughtNode]]] = {0: [[root_node]]}
        valid_nodes: Set[str] = {root_node.node_id}

        best_path = None
        final_evaluation = None

        for depth in range(self.tot.max_depth):
            current_level_paths = []

            # Create tasks for parallel thought generation
            thought_tasks = []
            for path in paths_by_depth[depth]:
                if path[-1].node_id in valid_nodes:
                    task = self.tot.generate_thoughts(content, path, regenerate)
                    thought_tasks.append((path, task))

            # Execute thought generation tasks in parallel
            if thought_tasks:
                results = await asyncio.gather(*(task for _, task in thought_tasks))
                for (path, _), thoughts in zip(thought_tasks, results):
                    for thought in thoughts:
                        new_path = path + [thought]
                        current_level_paths.append(new_path)

            if not current_level_paths:
                break

            evaluation = await self.tot.evaluate_level(current_level_paths, content)

            if evaluation:
                valid_nodes = {evaluation.best_path} | set(evaluation.neutral_paths)
                paths_by_depth[depth + 1] = [
                    p for p in current_level_paths if p[-1].node_id in valid_nodes
                ]

                best_path = next(
                    p
                    for p in current_level_paths
                    if p[-1].node_id == evaluation.best_path
                )
                final_evaluation = evaluation

        if best_path and final_evaluation:
            self.tot.mark_best_path(best_path)
            return await self._make_final_decision(best_path, final_evaluation)

        raise ValueError("Failed to complete evaluation")

    async def _make_final_decision(
        self, path: List[ThoughtNode], evaluation: PathEvaluation
    ) -> PublishabilityDecision:
        path_content = "\n".join(
            [
                f"Analysis of {node.aspect}:\n{node.content}\n"
                for node in path
                if node.content != "root"
            ]
        )

        prompt = f"""Based on this complete analysis path and its evaluation:

    Reasoning Path:
    {path_content}

    Make a final decision about the paper's publishability. Consider all aspects analyzed
    and provide concrete recommendations for improvement or acceptance.
    """
        final_decision = await self.decision_llm.ainvoke(prompt)

        print("=======")
        print(f"Critic's evaluation rationale: {evaluation.rationale}")
        print("=======")
        print(f"Final Decision Response: {final_decision}")

        return final_decision
