from dotenv import load_dotenv
load_dotenv()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    # Modified to support contrastive evaluation
    best_path: str = Field(description="IDs of the single best reasoning path among all available paths")
    neutral_paths: List[str] = Field(description="IDs of acceptable paths that are neither best nor worst")
    pruned_paths: List[str] = Field(description="IDs of paths to be pruned out due to weak reasoning relative to others")
    rationale: str = Field(description="Reasoning for each path's categorization")

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

class EvaluationMode(Enum):
    SINGLE_PAPER = "single"
    REFERENCE_SET = "reference"
    TEST_SET = "test"

@dataclass
class EvaluationConfig:
    """Configuration for paper evaluation pipeline"""
    mode: EvaluationMode
    input_path: Path
    output_dir: Path
    max_papers: Optional[int] = None  # For test mode
    ai_content_threshold: float = 15.0  # Threshold for re-evaluation
    
    @classmethod
    def create_single_paper_config(cls, paper_path: str, output_dir: str) -> 'EvaluationConfig':
        return cls(
            mode=EvaluationMode.SINGLE_PAPER,
            input_path=Path(paper_path),
            output_dir=Path(output_dir)
        )
    
    @classmethod
    def create_reference_config(cls, reference_dir: str, output_dir: str) -> 'EvaluationConfig':
        return cls(
            mode=EvaluationMode.REFERENCE_SET,
            input_path=Path(reference_dir),
            output_dir=Path(output_dir)
        )
    
    @classmethod
    def create_test_config(cls, test_dir: str, output_dir: str, max_papers: int) -> 'EvaluationConfig':
        return cls(
            mode=EvaluationMode.TEST_SET,
            input_path=Path(test_dir),
            output_dir=Path(output_dir),
            max_papers=max_papers
        )

class LLMFactory:
    @staticmethod
    def create_llm(config: ModelConfig) -> BaseChatModel:
        base_llm = None
        if config.provider == ModelProvider.GROQ:
            base_llm = ChatGroq(
                model=config.model_name,
                temperature=config.temperature,
                max_retries=config.max_retries
            )
        elif config.provider == ModelProvider.GOOGLE:
            base_llm = ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                max_retries=config.max_retries
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
                max_retries=config.max_retries
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        return base_llm
        

class ThoughtNode:
    def __init__(self, content: str, aspect: str, node_id: str = None):
        self.content: str = content
        self.aspect: str = aspect
        self.children: List['ThoughtNode'] = []
        self.evaluation: Optional[str] = None
        self.node_id: str = node_id or str(uuid.uuid4())
        self.parent_id: Optional[str] = None

class TreeVisualizer:
    def __init__(self):
        self.G = nx.DiGraph()
        self.best_path_nodes = set()
        
    def add_node(self, node: ThoughtNode, parent_id: Optional[str] = None, is_best_path: bool = False):
        try:
            # Ensure we have valid content and node ID
            if not node or not node.node_id:
                print(f"Warning: Invalid node or node ID")
                return
                
            aspect = str(node.aspect) if node.aspect else "Unknown aspect"
            content = str(node.content) if node.content else "No content"
            
            # Create a readable label with proper truncation
            content_preview = textwrap.fill(content[:100], width=30)
            label = f"{aspect}\n{content_preview}"
            
            # Add node if it doesn't exist
            if node.node_id not in self.G:
                self.G.add_node(node.node_id, 
                            label=label,
                            evaluation=node.evaluation or "Not evaluated")
            
            # Update node attributes
            self.G.nodes[node.node_id]['is_best_path'] = is_best_path
            
            # Add edge if parent exists
            if parent_id:
                if not self.G.has_edge(parent_id, node.node_id):
                    self.G.add_edge(parent_id, node.node_id)
                    
            if is_best_path:
                self.best_path_nodes.add(node.node_id)
                
        except Exception as e:
            print(f"Error adding node to visualization: {str(e)}")
            print(f"Node details - ID: {node.node_id}, aspect: {node.aspect}")
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        path = []
        current = node_id
        while current in self.G:
            path.append(current)
            predecessors = list(self.G.predecessors(current))
            if not predecessors:
                break
            current = predecessors[0]
        return path
    
    def visualize(self, output_path: str = None):
        plt.close('all')
        fig = plt.figure(figsize=(20, 12))
        success = False
        
        try:
            if not self.G.nodes():
                print("Warning: No nodes in graph to visualize")
                return False
            
            # Find all nodes and ensure they're connected
            all_nodes = set(self.G.nodes())
            print(f"Total nodes in graph: {len(all_nodes)}")
            
            # Find root(s)
            roots = [n for n in all_nodes if not list(self.G.predecessors(n))]
            if not roots:
                print("Warning: No root node found in graph")
                return False
            
            root = roots[0]
            print(f"Using root node: {root}")
            
            # Calculate levels using BFS
            levels = {}
            queue = [(root, 0)]
            visited = {root}  # Track visited nodes
            
            while queue:
                node, level = queue.pop(0)
                levels[node] = level
                
                # Add all unvisited successors to queue
                for successor in self.G.successors(node):
                    if successor not in visited:
                        queue.append((successor, level + 1))
                        visited.add(successor)
            
            # Handle any disconnected nodes by assigning them to level 0
            disconnected = all_nodes - visited
            if disconnected:
                print(f"Warning: Found {len(disconnected)} disconnected nodes")
                for node in disconnected:
                    levels[node] = 0
            
            # Calculate x positions for nodes at each level
            x_positions = {}
            y_positions = {}
            for level in set(levels.values()):
                nodes_at_level = [n for n, l in levels.items() if l == level]
                width = len(nodes_at_level)
                for i, node in enumerate(nodes_at_level):
                    x_positions[node] = (i - (width - 1) / 2) * 2  # Multiply by 2 for more spacing
                    y_positions[node] = -level * 2  # Multiply by 2 for more spacing
            
            # Combine positions
            pos = {node: (x_positions[node], y_positions[node]) for node in all_nodes}
            
            # Draw nodes
            node_colors = ['lightgreen' if node in self.best_path_nodes else 'lightblue' 
                        for node in self.G.nodes()]
            
            nx.draw_networkx_nodes(self.G, pos,
                                node_color=node_colors,
                                node_size=4000)
            
            # Draw edges
            nx.draw_networkx_edges(self.G, pos,
                                edge_color='gray',
                                arrows=True,
                                arrowsize=20,
                                width=2,
                                connectionstyle="arc3,rad=0.2")
            
            # Draw labels
            labels = {
                node: self.G.nodes[node].get('label', '')
                for node in self.G.nodes()
            }
            
            nx.draw_networkx_labels(self.G, pos,
                                labels=labels,
                                font_size=6,
                                font_weight='bold')
            
            plt.title("Reasoning Tree Analysis\nGreen: Best Path, Blue: Alternative Thoughts",
                    fontsize=14,
                    pad=20)
            
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='lightgreen',
                        markersize=15,
                        label='Best Path'),
                plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='lightblue',
                        markersize=15,
                        label='Alternative Thoughts')
            ]
            plt.legend(handles=legend_elements,
                    loc='upper right',
                    fontsize=10)
            
            plt.tight_layout()
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"Saving visualization to {output_path}")
                plt.savefig(output_path,
                        bbox_inches='tight',
                        dpi=300,
                        format='png')
                success = os.path.exists(output_path)
                if not success:
                    print(f"Error: Failed to save visualization to {output_path}")
            else:
                success = True
            
            return success
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            plt.close(fig)


class TreeOfThoughts:
    def __init__(self, reasoning_llm: BaseChatModel, critic_llm: BaseChatModel,
                 max_branches: int = 3, max_depth: int = 3):
        self.reasoning_llm = reasoning_llm.with_structured_output(ReasoningThoughts)
        self.critic_llm = critic_llm.with_structured_output(PathEvaluation)
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.visualizer = TreeVisualizer()
        self.all_thoughts: Dict[str, ThoughtNode] = {}
        self.best_path_ids = set()

    async def generate_thoughts(self, context: str, current_path: List[ThoughtNode], regenerate: bool = False) -> List[ThoughtNode]:
        prompt = self._create_reasoning_prompt(context, current_path, regenerate)
        response = await self.reasoning_llm.ainvoke(prompt)
        
        parent_node = current_path[-1] if current_path else None
        thoughts = []
        
        for thought in response.thoughts[:self.max_branches]:
            node = ThoughtNode(
                content=thought,
                aspect=response.next_aspect,
                node_id=str(uuid.uuid4())
            )
            if parent_node:
                node.parent_id = parent_node.node_id
                parent_node.children.append(node)
                
            self.all_thoughts[node.node_id] = node
            self.visualizer.add_node(node, parent_node.node_id if parent_node else None)
            thoughts.append(node)
        
        return thoughts

    async def evaluate_level(self, paths: List[List[ThoughtNode]], paper_content: str) -> Optional[PathEvaluation]:
        if not paths:
            return None
        
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                prompt = self._create_critic_prompt(paths, paper_content)
                evaluation = await self.critic_llm.ainvoke(prompt)
                
                if not evaluation or not isinstance(evaluation, PathEvaluation):
                    print("Warning: Invalid evaluation response from critic LLM")
                    retries += 1
                    if retries < max_retries:
                        print(f"Retrying evaluation (attempt {retries + 1}/{max_retries})...")
                        continue
                    raise ValueError("Failed to get valid evaluation from critic LLM after maximum retries")
                
                if not evaluation.best_path or not evaluation.pruned_paths:
                    print("Warning: Missing required fields in evaluation")
                    retries += 1
                    if retries < max_retries:
                        print(f"Retrying evaluation (attempt {retries + 1}/{max_retries})...")
                        continue
                    raise ValueError("Failed to get complete evaluation fields after maximum retries")
                
                for path in paths:
                    node = path[-1]
                    if node.node_id == evaluation.best_path:
                        node.evaluation = "best"
                    elif node.node_id in evaluation.pruned_paths:
                        node.evaluation = "pruned"
                    elif node.node_id in evaluation.neutral_paths:
                        node.evaluation = "neutral"
                    else:
                        node.evaluation = "unknown"
                        
                return evaluation
                
            except Exception as e:
                print(f"Error in evaluate_level: {str(e)}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying evaluation (attempt {retries + 1}/{max_retries})...")
                    continue
                raise

    def mark_best_path(self, path: List[ThoughtNode]):
        """Mark nodes that are part of the best reasoning path"""
        self.best_path_ids = {node.node_id for node in path}
        
        # Update visualization
        for node_id in self.all_thoughts:
            self.visualizer.add_node(
                self.all_thoughts[node_id],
                self.all_thoughts[node_id].parent_id,
                is_best_path=(node_id in self.best_path_ids)
            )

    def _create_reasoning_prompt(self, context: str, current_path: List[ThoughtNode], regenerate: bool = False) -> str:
        path_summary = "\n".join([
            f"- Analyzing {node.aspect}: {node.content}" 
            for node in current_path
        ])
        
        if not regenerate:
            print("Using thought gen...")
            return f"""Analyze this research paper's publishability through careful reasoning.

    Previous analysis steps:
    {path_summary if current_path else "No previous analysis"}

    Paper excerpt:
    {context}

    Based on the previous analysis (if any), YOUR TASK is TO PROVIDE {self.max_branches} DISTINCT reasoning thoughts about different aspects of the paper's publishability. Each thought should be thorough, well-supported, AND explore new dimensions or deepen previous analysis.

    NOTE: you may not have memory of it, but you have already undergone the task of generating rich thoughts which, on analysis by an expert determine if the paper deems to be publishable or not. However you werent correct in your thought generation and have been given a final chance, so work well else, I may be fired from my job. So, ensure critically ahdering to all the crieterion given and generate, rich, detailed, diverse thoughts that deeply underrstand multiple aspects of the paper. Weighed on your thoughts' observations and intuitions about the different aspects of the given paper, the user will determine if your thinking indicates it might be publishable or non-publishable, hence ensure rich and critically thought out thoughts. 

    For potentially unpublishable papers, examine:
    - Logical inconsistencies and contradictions in methodology or results
    - Lack of scientific rigor or proper controls
    - Ambiguity in reasoning or conclusions
    - Flamboyant grammar, unrealisitc claims, or overstatements
    - Missing or inadequate statistical analysis
    - Inappropriate writing style or structure
    - Unsupported claims or conclusions
    - Major organizational issues
    - Ethical concerns or red flags
    - Mismatch in the topics or ideations between different parts of the paper.
    - Relevant content in few sections, but very shortly or poorly written content in other sections

    For potentially publishable papers, analyze:
    - Novel contributions to the field
    - Coherence of ideas across the paper
    - No other worldly or over exaggerated ideations or claims
    - Novelty, if any
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
    1. Focus on a specific aspect or a few different aspects
    2. Provide concrete evidence from the text
    3. Evaluate impact on publishability
    4. Connect to previous reasoning when relevant
    5. Suggest potential improvements if applicable

    Ensure thoughts are diverse(ie are well-varied), non-repetitive(ie dont repeat previously said reasonings) and explore varied dimensions of the paper's quality."""

        else: 
            print("FAILED: Using thought regen")
            return f"""Analyze this research paper's publishability through careful reasoning.

    Previous analysis steps:
    {path_summary if current_path else "No previous analysis"}

    Paper excerpt:
    {context}

    Based on the previous analysis (if any), YOUR TASK is TO PROVIDE {self.max_branches} DISTINCT reasoning thoughts about different aspects of the paper's publishability. Each thought should be thorough, well-supported, and explore a new dimension and deepen previous analysis.

    NOTE: You may not have memory of it, but you have failed at making good thoughts in the past and due to that failure this is your last chance to make good thoughts by following my instructions. It has been detected that the paper excerpt provided to you has a substantial amount of 'potentially' ai generated content and hence may not be a truly publishable paper. Hence, have a rich thought process that deeply understand different aspects of the paper via detailed and rich varied thoughts. Weighed on your thoughts' observations and intuitions about the different aspects of the given paper, the user will determine if your thinking indicates it might be publishable or non-publishable. Even despite potentially having ai content, this paper still can be deemed publishable. But it all depends on your thought process. Hence ensure varied and rich diverse thoughts by adhering the guidelines given to you as follows as well as by using your intellect.

    For potentially unpublishable papers, examine:
    - Logical inconsistencies and contradictions in methodology or results
    - Lack of scientific rigor or proper controls
    - Ambiguity in reasoning or conclusions
    - Flamboyant grammar, unrealisitc claims, or overstatements
    - Missing or inadequate statistical analysis
    - Inappropriate writing style or structure
    - Unsupported claims or conclusions
    - Major organizational issues
    - Ethical concerns or red flags
    - Mismatch in the topics or ideations between different parts of the paper.
    - Relevant content in few sections, but very shortly or poorly written content in other sections

    For potentially publishable papers, analyze:
    - Novel contributions to the field
    - Coherence of ideas across the paper
    - No other worldly or over exaggerated ideations or claims
    - Novelty, if any
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
    1. Focus on a specific aspect or a few different aspects
    2. Provide concrete evidence from the text
    3. Evaluate impact on publishability
    4. Connect to previous reasoning when relevant
    5. Suggest potential improvements if applicable

    Ensure thoughts are diverse(ie are well-varied), non-repetitive(ie dont repeat previously said reasonings) and explore varied dimensions of the paper's quality."""

    def _create_critic_prompt(self, paths: List[List[ThoughtNode]], paper_content: str) -> str:
        num_to_prune = max(1, len(paths) // 3)
        paths_content = []
        for i, path in enumerate(paths):
            path_steps = [f"Step {j+1} ({node.aspect}): {node.content}" 
                        for j, node in enumerate(path)]
            paths_content.append(f"Path {i+1} (ID: {path[-1].node_id}):\n" + 
                            "\n".join(path_steps))
                
        return f"""You are evaluating multiple reasoning paths about a research paper's publishability.

    Paper Content:
    {paper_content}  

    Reasoning Paths to Evaluate:
    {"\n\n".join(paths_content)}

    Compare these paths using detailed contrastive analysis. For each path verify against the paper content and consider:

1. Logical Strength:
   - Coherence and flow between reasoning steps
   - Quality of evidence cited from the paper
   - Depth of analysis at each step
   - Connections made between different aspects

2. Coverage & Insight:
   - Range of critical aspects examined
   - Novel observations and insights
   - Integration of multiple paper elements
   - Potential for valuable further analysis

3. Scientific Rigor:
   - Attention to methodology
   - Statistical reasoning where relevant
   - Treatment of limitations and assumptions
   - Consideration of alternative explanations

4. Relative Merits:
   - Direct comparisons with other paths
   - Unique strengths and weaknesses
   - Value added beyond other paths
   - Gaps or oversights compared to others 

REQUIRED OUTPUT STRUCTURE:
1. Select EXACTLY ONE path as best (must choose single strongest)
2. Prune EXACTLY {num_to_prune} paths (choose relatively weakest)
3. Mark remaining {len(paths) - num_to_prune - 1} paths as neutral

Provide your rationale as a cohesive analysis explaining:
- Why you selected the best path and how it compares to others
- Why certain paths were marked for pruning
- The relative merits of neutral paths
- How the paths differ in their analysis quality and coverage

These proportions are strict requirements - exactly one best, {num_to_prune} pruned, rest neutral.

ENSURE TO ADHERE TO THE GIVEN STRUCTURED OUTPUT FORMAT AS YOUR RESPONSE
"""



class PaperEvaluator:
    def __init__(self, reasoning_config: ModelConfig, critic_config: ModelConfig):
        reasoning_llm = LLMFactory.create_llm(reasoning_config)
        critic_llm = LLMFactory.create_llm(critic_config)
        
        self.tot = TreeOfThoughts(reasoning_llm, critic_llm)
        self.decision_llm = critic_llm.with_structured_output(PublishabilityDecision)

    async def evaluate_paper(self, content: str, regenerate: bool = False) -> PublishabilityDecision:
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
                paths_by_depth[depth + 1] = [p for p in current_level_paths 
                                        if p[-1].node_id in valid_nodes]
                
                best_path = next(p for p in current_level_paths 
                            if p[-1].node_id == evaluation.best_path)
                final_evaluation = evaluation
        
        if best_path and final_evaluation:
            self.tot.mark_best_path(best_path)
            return await self._make_final_decision(best_path, final_evaluation)
        
        raise ValueError("Failed to complete evaluation")

    async def _make_final_decision(
        self,
        path: List[ThoughtNode],
        evaluation: PathEvaluation
    ) -> PublishabilityDecision:
        path_content = "\n".join([
            f"Analysis of {node.aspect}:\n{node.content}\n"
            for node in path if node.content != "root"
        ])
        
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



def extract_pdf_content(pdf_path: str) -> str:
    """
    Extract content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text content from the PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    try:
        # Extract content using partition_pdf
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            pdf_infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters= 2000,
        )
        
        # Combine all text elements
        content = ""
        for element in elements:
            content += element.text
            
        if not content.strip():
            raise ValueError("No content extracted from PDF")
            
        return content
        
    except Exception as e:
        raise Exception(f"Error extracting PDF content: {str(e)}")



async def check_content(content):
    detector = checker()
    print(f"Checking content of {content[:50]}...")
    results = await detector._run_async(content)  # Assuming _run_async exists
    return results

class PaperEvaluationPipeline:
    def __init__(self, config: EvaluationConfig, evaluator: PaperEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.results = []
        
    async def run(self) -> pd.DataFrame:
        """Run the evaluation pipeline based on configured mode"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.mode == EvaluationMode.SINGLE_PAPER:
            result = await self._evaluate_single_paper(self.config.input_path)
            self.results.append(result)
            
        elif self.config.mode == EvaluationMode.REFERENCE_SET:
            await self._evaluate_reference_set()
            
        elif self.config.mode == EvaluationMode.TEST_SET:
            await self._evaluate_test_set()
            
        df = pd.DataFrame(self.results)
        self._save_results(df)
        self._analyze_results(df)
        
        return df

    async def _evaluate_single_paper(self, paper_path: Path) -> Dict[str, Any]:
        try:
            content = extract_pdf_content(str(paper_path))
            ai_percentage = await check_content(content)
            
            decision = await self.evaluator.evaluate_paper(content)
            
            if (ai_percentage["average_fake_percentage"] > self.config.ai_content_threshold) and decision.is_publishable:
                print(f"High AI content detected ({ai_percentage}%). Re-evaluating...")
                decision = await self.evaluator.evaluate_paper(content, regenerate=True)
            
            paper_output_dir = self.config.output_dir / paper_path.stem
            paper_output_dir.mkdir(parents=True, exist_ok=True)
            
            viz_path = paper_output_dir / f"{paper_path.stem}_thought_tree.png"
            self.evaluator.tot.visualizer.visualize(str(viz_path))
            
            result = {
                'paper_id': paper_path.stem,
                'predicted_label': decision.is_publishable,
                'primary_strengths': '|'.join(decision.primary_strengths),
                'critical_weaknesses': '|'.join(decision.critical_weaknesses),
                'recommendation': decision.recommendation,
                'file_path': str(paper_path),
                'output_dir': str(paper_output_dir),
                'ai_content_percentage': ai_percentage
            }
            
            self._save_paper_analysis(paper_output_dir, result, decision)
            
            return result
            
        except Exception as e:
            print(f"Error processing {paper_path}: {str(e)}")
            return {
                'paper_id': paper_path.stem,
                'predicted_label': None,
                'primary_strengths': '',
                'critical_weaknesses': '',
                'recommendation': f'ERROR: {str(e)}',
                'file_path': str(paper_path),
                'output_dir': '',
                'ai_content_percentage': None
            }
    
    async def _evaluate_reference_set(self):
        """Evaluate papers in reference set structure"""
        # Process Non-Publishable papers
        non_pub_dir = self.config.input_path / "Non-Publishable"
        if non_pub_dir.exists():
            for paper_path in non_pub_dir.glob("*.pdf"):
                result = await self._evaluate_single_paper(paper_path)  # Add await
                result['true_label'] = False
                result['conference'] = 'Non-Publishable'
                result['correct_prediction'] = result['predicted_label'] == False
                self.results.append(result)
        
        # Process Publishable papers
        pub_dir = self.config.input_path / "Publishable"
        if pub_dir.exists():
            for conf_dir in pub_dir.iterdir():
                if conf_dir.is_dir():
                    for paper_path in conf_dir.glob("*.pdf"):
                        result = await self._evaluate_single_paper(paper_path)  # Add await
                        result['true_label'] = True
                        result['conference'] = conf_dir.name
                        result['correct_prediction'] = result['predicted_label'] == True
                        self.results.append(result)

    async def _evaluate_test_set(self):
        """Evaluate papers in test set"""
        if not self.config.input_path.exists():
            raise ValueError(f"Test directory does not exist: {self.config.input_path}")
            
        paper_paths = sorted(list(self.config.input_path.glob("*.pdf")))
        
        if self.config.max_papers:
            paper_paths = paper_paths[:self.config.max_papers]
            
        for paper_path in paper_paths:
            result = await self._evaluate_single_paper(paper_path)  # Add await
            self.results.append(result)
    
    def _save_paper_analysis(self, output_dir: Path, result: Dict[str, Any], decision: PublishabilityDecision):
        """Save detailed analysis for a paper"""
        # Save tree data as JSON
        tree_data = {
            "metadata": {
                "paper_path": result['file_path'],
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "analysis_result": decision.is_publishable,
                "ai_content_percentage": result['ai_content_percentage']
            },
            "nodes": [
                {
                    "id": node_id,
                    "content": node.content,
                    "aspect": node.aspect,
                    "evaluation": node.evaluation,
                    "parent": node.parent_id
                }
                for node_id, node in self.evaluator.tot.all_thoughts.items()
            ],
            "final_decision": {
                "is_publishable": decision.is_publishable,
                "primary_strengths": decision.primary_strengths,
                "critical_weaknesses": decision.critical_weaknesses,
                "recommendation": decision.recommendation
            }
        }
        
        json_path = output_dir / f"{result['paper_id']}_thought_tree_data.json"
        with open(json_path, 'w') as f:
            json.dump(tree_data, f, indent=2)
            
        # Create analysis summary
        summary_path = output_dir / f"{result['paper_id']}_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Paper Analysis Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Paper: {result['file_path']}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Publishable: {decision.is_publishable}\n")
            f.write(f"AI Content Percentage: {result['ai_content_percentage']}%\n\n")
            f.write("Key Strengths:\n")
            for strength in decision.primary_strengths:
                f.write(f"- {strength}\n")
            f.write("\nCritical Weaknesses:\n")
            for weakness in decision.critical_weaknesses:
                f.write(f"- {weakness}\n")
            f.write(f"\nRecommendation:\n{decision.recommendation}\n")
    
    def _save_results(self, df: pd.DataFrame):
        """Save evaluation results to CSV"""
        csv_path = self.config.output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    def _analyze_results(self, df: pd.DataFrame):
        """Analyze and print evaluation results"""
        total_papers = len(df)
        valid_predictions = df['predicted_label'].notna()
        df_valid = df[valid_predictions]
        
        if len(df_valid) == 0:
            print("\nNo valid predictions to analyze!")
            return
            
        if 'correct_prediction' in df_valid.columns:
            correct_predictions = df_valid['correct_prediction'].sum()
            accuracy = (correct_predictions / len(df_valid)) * 100
            
            print("\nEvaluation Results Analysis:")
            print(f"Total papers processed: {total_papers}")
            print(f"Papers with valid predictions: {len(df_valid)}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.2f}%")
            
            # Confusion matrix for valid predictions
            if 'true_label' in df_valid.columns:
                true_pos = len(df_valid[(df_valid['true_label'] == True) & (df_valid['predicted_label'] == True)])
                true_neg = len(df_valid[(df_valid['true_label'] == False) & (df_valid['predicted_label'] == False)])
                false_pos = len(df_valid[(df_valid['true_label'] == False) & (df_valid['predicted_label'] == True)])
                false_neg = len(df_valid[(df_valid['true_label'] == True) & (df_valid['predicted_label'] == False)])
                
                print("\nConfusion Matrix:")
                print(f"True Positives: {true_pos}")
                print(f"True Negatives: {true_neg}")
                print(f"False Positives: {false_pos}")
                print(f"False Negatives: {false_neg}")
        
        # Print failed processing cases
        failed_cases = df[~valid_predictions]
        if len(failed_cases) > 0:
            print("\nFailed Processing Cases:")
            for _, row in failed_cases.iterrows():
                print(f"- {row['paper_id']}: {row['recommendation']}")


async def main():

    start = time.time()
    reasoning_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini"
    )
    
    critic_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o"
    )
    
    evaluator = PaperEvaluator(reasoning_config, critic_config)
    
        
    # 1. Single paper evaluation
    config = EvaluationConfig.create_single_paper_config(
        paper_path="/home/divyansh/code/kdsh/dataset/Papers/P013.pdf",
        output_dir="single_paper_analysis"
    )
    
    # 2. Reference set evaluation
    # config = EvaluationConfig.create_reference_config(
    #     reference_dir="/path/to/reference",
    #     output_dir="reference_analysis"
    # )
    
    # # 3. Test set evaluation
    # config = EvaluationConfig.create_test_config(
    #     test_dir="/path/to/test/papers",
    #     output_dir="test_analysis",
    #     max_papers=50
    # )
    
    # Run evaluation pipeline
    pipeline = PaperEvaluationPipeline(config, evaluator)

    results_df = await pipeline.run()
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())