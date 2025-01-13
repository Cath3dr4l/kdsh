from dotenv import load_dotenv

load_dotenv()

import sys
import os
import pandas as pd
from typing import List, Dict, Any, Optional
import csv
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import asyncio

# from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


# Pydantic models for structured outputs
class ReasoningThoughts(BaseModel):
    thoughts: List[str] = Field(
        description="List of reasoning steps about the paper's publishability based on previous context. "
        "Each thought should be a complete, logical analysis step."
    )
    next_aspect: str = Field(
        description="The next aspect of the paper to analyze based on current reasoning"
    )


class PathEvaluation(BaseModel):
    strength: str = Field(
        description="Qualitative assessment of the reasoning path's strength (e.g., 'very strong', 'needs improvement')"
    )
    rationale: str = Field(
        description="Detailed explanation of why this reasoning path is strong or weak"
    )


class PublishabilityDecision(BaseModel):
    is_publishable: bool = Field(
        description="Whether the paper is publishable based on the complete reasoning path"
    )
    primary_strengths: List[str] = Field(
        description="Key strengths that support publishability"
    )
    critical_weaknesses: List[str] = Field(
        description="Critical weaknesses that affect publishability"
    )
    recommendation: str = Field(
        description="Detailed publication recommendation including suggested improvements if any"
    )


class ModelProvider(Enum):
    GROQ = "groq"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENAI = "opeani"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    temperature: float = 0
    max_retries: int = 2


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


class ThoughtNode:
    def __init__(self, content: str, aspect: str):
        self.content: str = content
        self.aspect: str = aspect
        self.children: List["ThoughtNode"] = []
        self.evaluation: str = None


class TreeOfThoughts:
    def __init__(
        self,
        reasoning_llm: BaseChatModel,
        critic_llm: BaseChatModel,
        max_branches: int = 3,
        max_depth: int = 3,
    ):
        self.reasoning_llm = reasoning_llm.with_structured_output(ReasoningThoughts)
        self.critic_llm = critic_llm.with_structured_output(PathEvaluation)
        self.max_branches = max_branches
        self.max_depth = max_depth

    def _create_reasoning_prompt(
        self, context: str, current_path: List[ThoughtNode]
    ) -> str:
        path_summary = "\n".join(
            [f"- Analyzing {node.aspect}: {node.content}" for node in current_path]
        )

        return f"""Analyze this research paper's publishability through careful reasoning.

                    Previous analysis steps:
                    {path_summary if current_path else "No previous analysis"}

                    Paper excerpt:
                    {context}

                    Based on the previous analysis (if any), YOUR TASK is TO PROVIDE {self.max_branches} DISTINCT reasoning thoughts about different aspects of the paper's publishability. Each thought should be thorough, well-supported, and explore a new dimension or deepen previous analysis.

                    For potentially unpublishable papers, examine:
                    - Logical inconsistencies and contradictions in methodology or results
                    - Lack of scientific rigor or proper controls
                    - Missing or inadequate statistical analysis
                    - Inappropriate writing style or structure
                    - Unsupported claims or conclusions
                    - Major organizational issues
                    - Ethical concerns or red flags

                    For potentially publishable papers, analyze:
                    - Novel contributions to the field
                    - Methodological innovations
                    - Rigor of experimental design
                    - Quality of data analysis and visualization
                    - Clarity of writing and argumentation
                    - Theoretical foundations
                    - Practical implications

                    For all papers, consider:
                    - Internal consistency and coherence
                    - Quality of literature review and citations
                    - Adherence to scientific method
                    - Reproducibility of methods
                    - Data presentation and analysis
                    - Validity of conclusions
                    - Technical soundness
                    - Writing clarity and professionalism
                    - Alignment with journal standards

                    Each thought should:
                    1. Focus on a specific aspect
                    2. Provide concrete evidence from the text
                    3. Evaluate impact on publishability
                    4. Connect to previous reasoning when relevant
                    5. Suggest potential improvements if applicable

                    Ensure thoughts are diverse and explore varied dimensions of the paper's quality.
            """

    def _create_critic_prompt(self, path: List[ThoughtNode]) -> str:
        path_content = "\n".join(
            [
                f"Step {i+1} ({node.aspect}): {node.content}"
                for i, node in enumerate(path)
            ]
        )

        return f"""Evaluate this reasoning path about a research paper's publishability:

                    Reasoning Path:
                    {path_content}

                    Provide a qualitative assessment of this reasoning path's strength and a detailed rationale
                    for your evaluation. Consider:
                    - Logical flow and progression of thoughts
                    - Coverage of crucial aspects
                    - Depth of analysis
                    - Validity of conclusions
                """

    async def generate_thoughts(
        self, context: str, current_path: List[ThoughtNode]
    ) -> List[ThoughtNode]:
        prompt = self._create_reasoning_prompt(context, current_path)
        response = await self.reasoning_llm.ainvoke(prompt)

        return [
            ThoughtNode(content=thought, aspect=response.next_aspect)
            for thought in response.thoughts[: self.max_branches]
        ]

    async def evaluate_path(self, path: List[ThoughtNode]) -> PathEvaluation:
        prompt = self._create_critic_prompt(path)
        return await self.critic_llm.ainvoke(prompt)


class PaperEvaluator:
    def __init__(self, reasoning_config: ModelConfig, critic_config: ModelConfig):
        reasoning_llm = LLMFactory.create_llm(reasoning_config)
        critic_llm = LLMFactory.create_llm(critic_config)

        self.tot = TreeOfThoughts(reasoning_llm, critic_llm)
        self.decision_llm = critic_llm.with_structured_output(PublishabilityDecision)

    async def evaluate_paper(self, content: str) -> PublishabilityDecision:
        best_path = []
        best_evaluation = None

        async def explore_path(current_path: List[ThoughtNode], depth: int) -> None:
            nonlocal best_path, best_evaluation

            if depth >= self.tot.max_depth:
                return

            thoughts = await self.tot.generate_thoughts(content, current_path)

            # Gather evaluations concurrently
            evaluations = await asyncio.gather(
                *(
                    self.tot.evaluate_path(current_path + [thought])
                    for thought in thoughts
                )
            )

            for thought, evaluation in zip(thoughts, evaluations):
                if not best_evaluation or self._is_better_evaluation(
                    evaluation, best_evaluation
                ):
                    best_path = current_path + [thought]
                    best_evaluation = evaluation

                await explore_path(current_path + [thought], depth + 1)

        # Start exploration
        await explore_path([], 0)

        # Make final decision
        return await self._make_final_decision(best_path, best_evaluation)

    def _is_better_evaluation(
        self, eval1: PathEvaluation, eval2: PathEvaluation
    ) -> bool:
        # Simple heuristic - can be made more sophisticated
        strong_indicators = ["very strong", "excellent", "comprehensive"]
        weak_indicators = ["weak", "insufficient", "poor"]

        return any(
            ind in eval1.strength.lower() for ind in strong_indicators
        ) and not any(ind in eval1.strength.lower() for ind in weak_indicators)

    async def _make_final_decision(
        self, path: List[ThoughtNode], evaluation: PathEvaluation
    ) -> PublishabilityDecision:
        path_content = "\n".join(
            [f"Analysis of {node.aspect}:\n{node.content}\n" for node in path]
        )

        prompt = f"""Based on this complete analysis path and its evaluation:

                    Reasoning Path:
                    {path_content}
                    
                    Evaluation:
                    Strength: {evaluation.strength}
                    Rationale: {evaluation.rationale}
                    
                    Make a final decision about the paper's publishability. Consider all aspects analyzed
                    and provide concrete recommendations.
                """
        return await self.decision_llm.ainvoke(prompt)


# from PyPDF2 import PdfReader


def extract_pdf_content_with_pypdf2(relative_path: str) -> str:
    """
    Extract content from a PDF file using PyPDF2.

    Args:
        relative_path: Relative path to the PDF file

    Returns:
        Extracted text content from the PDF
    """
    # Get the absolute path
    pdf_path = os.path.abspath(relative_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Read the PDF file
        reader = PdfReader(pdf_path)
        content = ""

        # Extract text from each page
        for page in reader.pages:
            content += page.extract_text() or ""

        if not content.strip():
            raise ValueError("No content extracted from PDF")

        return content

    except Exception as e:
        raise Exception(f"Error extracting PDF content: {str(e)}")


def main():
    # Configure models
    reasoning_config = ModelConfig(
        provider=ModelProvider.OPENAI, model_name="gpt-4o-mini"
    )

    critic_config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4o-mini")

    print("Configuring evaluator...")

    evaluator = PaperEvaluator(reasoning_config, critic_config)
    pdf_path = "../dataset/Papers/P001.pdf"
    # content = extract_pdf_content_with_pypdf2(pdf_path)
    content = """
    Leveraging Clustering Techniques for Enhanced
Drone Monitoring and Position Estimation
Abstract
Drone tracking and localization are essential for various applications, including
managing drone formations and implementing anti-drone strategies. Pinpointing
and monitoring drones in three-dimensional space is difficult, particularly when
trying to capture the subtle movements of small drones during rapid maneuvers.
This involves extracting faint signals from varied flight settings and maintaining
alignment despite swift actions. Typically, cameras and LiDAR systems are used
to record the paths of drones. However, they encounter challenges in categorizing
drones and estimating their positions accurately. This report provides an overview
of an approach named CL-Det. It uses a clustering-based learning detection strategy
to track and estimate the position of drones using data from two types of LiDAR
sensors: Livox Avia and LiDAR 360. This method merges data from both LiDAR
sources to accurately determine the drone’s location in three dimensions. The
method begins by synchronizing the time codes of the data from the two sensors
and then isolates the point cloud data for the objects of interest (OOIs) from the
environmental data. A Density-Based Spatial Clustering of Applications with
Noise (DBSCAN) method is applied to cluster the OOI point cloud data, and the
center point of the most prominent cluster is taken as the drone’s location. The
technique also incorporates past position estimates to compensate for any missing
information.
1 Introduction
Unmanned aerial vehicles (UAVs), commonly referred to as drones, have gained prominence and
significantly influence areas like logistics, imaging, and emergency response, offering substantial
advantages to society. However, the broad adoption and sophisticated features of compact, off-the-
shelf drones have created intricate security issues that extend beyond conventional risks.
Recent years have witnessed a surge in research on anti-UAV systems. Present anti-UAV methods
predominantly utilize visual, radar, and radio frequency (RF) technologies. Despite these strides,
recognizing drones poses a considerable hurdle for sensors like cameras, particularly when drones
are at significant altitudes or in challenging visual environments. These methods usually fail to spot
small drones because of their minimal size, which leads to a decreased radar cross-section and a
less noticeable visual presence. Furthermore, current anti-UAV studies primarily focus on detecting
objects and tracking them in two dimensions, overlooking the crucial element of estimating their
3D paths. This omission significantly restricts the effectiveness of anti-UAV systems in practical,
real-world contexts.
Our proposed solution, a detection method based on clustering learning (CL-Det), uses the strengths
of both Livox Avia and LiDAR 360 to improve the tracking and position estimation of UAVs.
Initially, the timestamps from the Livox Avia and LiDAR 360 data are aligned to maintain temporal
consistency. By examining the LiDAR data, which contains the spatial coordinates of objects at
specific times, and comparing these to the actual recorded positions of the drone at those times, the
drone’s location within the LiDAR point cloud data is effectively pinpointed. The point cloud for
.
objects of interest (OOIs) is then isolated from the environmental data. The point cloud of the OOIs
is grouped using the DBSCAN algorithm, and the central point of the largest cluster is designated as
the UAV’s position. Moreover, radar data also faces significant challenges due to missing information.
To mitigate potential data deficiencies, past estimations are employed to supplement missing data,
thereby maintaining the consistency and precision of UAV tracking.
2 Methodology
This section details the methodology employed to ascertain the drone’s spatial position utilizing
information from LiDAR 360 and Livox Avia sensors. The strategy integrates data from both sensor
types to achieve precise position calculations.
2.1 Data Sources
The following modalities of data were utilized:
• Double fisheye camera visual images
• Livox Mid-360 (LiDAR 360) 3D point cloud data
• Livox Avia 3D point cloud data
• Millimeter-wave radar 3D point cloud data
Only 14 out of 59 test sequences have non-zero radar values; therefore, the radar dataset is excluded
from this work due to data availability issues. Two primary sensor types are employed: LiDAR 360
and Livox Avia, both of which supply 3D point cloud data crucial for identifying the drone’s location.
The detailed data descriptions are outlined as follows:
• LiDAR 360 offers a complete 360-degree view with 3D point cloud data. This dataset
encompasses environmental details and other observable objects.
• Livox Avia delivers focused 3D point cloud data at specific timestamps, typically indicating
the origin point or the drone’s position.
2.2 Algorithm
For every sequence, corresponding positions are recorded at specific timestamps. The procedure
gives precedence to LiDAR 360 data, using Livox Avia data as a backup if the former is not available.
If neither source is accessible, the position is estimated using historical averages.
2.2.1 LiDAR 360 Data Processing
• Separation of Points: The LiDAR 360 data is visually examined to classify areas into two
zones: environment and non-environment zones.
• Removal of Environment Points: All points within the environment zone are deemed part
of the surroundings and are thus excluded from the dataset. After removing environment
points, it is observed that the remaining non-environment points imply the drone position.
• Clustering: The DBSCAN clustering algorithm is applied to the remaining points to discern
distinct clusters.
• Cluster Selection: The most extensive non-environment cluster is chosen as the representa-
tive group of points that correspond to the drone.
• Mean Position Calculation: The drone’s position is determined by calculating the mean of
the selected cluster, represented by (x, y, z) coordinates.
2.2.2 Livox Avia Data Processing
• Removal of Noise: Points with coordinates (0, 0, 0) are eliminated as they are regarded as
noise.
• Mean Position Calculation: The mean of the residual points is computed to ascertain the
drone’s position in (x, y, z) coordinates.
2
2.2.3 Fallback Method
When neither LiDAR 360 nor Livox Avia data is available, the average location of the drone derived
from training datasets is used. The average ground truth position (x, y, z) from all training datasets
estimates the drone ground truth position, which is (0.734, -9.739, 33.353).
2.3 Implementation Details
The program fetches LiDAR 360 or Livox Avia data from the nearest timestamp for each sequence,
as indicated in the test dataset. Clustering is executed using the DBSCAN algorithm with appro-
priate parameters to guarantee strong clustering. Visual inspection is employed for the preliminary
separation of points, ensuring an accurate categorization of environment points.
The implementation was conducted on a Lenovo IdeaPad Slim 5 Pro (16") running Windows 11
with an AMD Ryzen 7 5800H CPU and 16GB DDR4 RAM. The analysis was carried out in a
Jupyter Notebook environment using Python 3.10. For clustering, the DBSCAN algorithm from the
Scikit-Learn library was utilized. The DBSCAN algorithm was configured with an epsilon (eps)
value of 2 and a minimum number of points (minPts) set to 1.
3 Results
The algorithm achieved a pose MSE loss of 120.215 and a classification accuracy of 0.322. Table 1
presents the evaluation results compared to other teams.
Table 1: Evaluation results on the leaderboard
Team ID Pose MSE (↓) Accuracy (↑)
SDUCZS 58198 2.21375 0.8136
Gaofen Lab 57978 7.299575 0.3220
sysutlt 57843 24.50694 0.3220
casetrous 58233 56.880267 0.2542
NTU-ICG (ours) 58268 120.215107 0.3220
MTC 58180 189.669428 0.2724
gzist 56936 417.396317 0.2302
4 Conclusions
This paper introduces a clustering-based learning method, CL-Det, which employs advanced cluster-
ing techniques such as K-Means and DBSCAN for drone detection and position estimation using
LiDAR data. The approach guarantees dependable and precise drone position estimation by utilizing
multi-sensor data and robust clustering methods. Fallback mechanisms are in place to ensure con-
tinuous position estimation even when primary sensor data is absent. Through thorough parameter
optimization and comparative assessment, the proposed method’s effective performance in drone
tracking and position estimation is demonstrated.
    """
    decision = asyncio.run(evaluator.evaluate_paper(content=content))
    print(decision)


if __name__ == "__main__":
    main()
