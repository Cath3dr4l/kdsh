from dotenv import load_dotenv
load_dotenv()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import List, Dict, Any, Optional, Set
import csv
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import textwrap

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
        self.best_path_nodes = set()  # Track nodes in the best path
        
    def add_node(self, node: ThoughtNode, parent_id: Optional[str] = None, is_best_path: bool = False):
        try:
            # Ensure we have valid content
            aspect = str(node.aspect) if node.aspect else "Unknown aspect"
            content = str(node.content) if node.content else "No content"
            
            # Create a readable label with proper truncation
            content_preview = textwrap.fill(content[:100], width=30)  # Show more content, wrapped
            label = f"{aspect}\n{content_preview}"
            
            self.G.add_node(node.node_id, 
                        label=label,
                        evaluation=node.evaluation or "Not evaluated",
                        depth=len(self.get_path_to_root(node.node_id)),
                        is_best_path=is_best_path)
            
            if parent_id:
                self.G.add_edge(parent_id, node.node_id)
                
            if is_best_path:
                self.best_path_nodes.add(node.node_id)
                
        except Exception as e:
            print(f"Error adding node to visualization: {str(e)}")
            print(f"Node details - aspect: {node.aspect}, content length: {len(node.content) if node.content else 0}")
    
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
                raise ValueError("No nodes in graph to visualize")
            
            # Use hierarchical layout for top-down tree visualization
            pos = nx.kamada_kawai_layout(self.G)
            
            root = [n for n in self.G.nodes() if not list(self.G.predecessors(n))][0]
            
            levels = {}
            queue = [(root, 0)]
            visited = set()
            
            while queue:
                node, level = queue.pop(0)
                if node not in visited:
                    levels[node] = level
                    visited.add(node)
                    # Add all successors to queue with incremented level
                    for successor in self.G.successors(node):
                        queue.append((successor, level + 1))
            
            # Calculate x coordinates to spread nodes at each level
            x_positions = {}
            for level in set(levels.values()):
                nodes_at_level = [n for n, l in levels.items() if l == level]
                width = len(nodes_at_level) - 1
                for i, node in enumerate(sorted(nodes_at_level)):
                    x_positions[node] = (i - width/2) if width > 0 else 0
            
            # Create positions dictionary with calculated coordinates
            pos = {node: (x_positions[node], -levels[node]) for node in self.G.nodes()}
            
            # Draw nodes
            node_colors = ['lightgreen' if node in self.best_path_nodes else 'lightblue' 
                        for node in self.G.nodes()]
            
            nx.draw_networkx_nodes(self.G, pos,
                                node_color=node_colors,
                                node_size=4000)
            
            # Draw edges with curved arrows
            nx.draw_networkx_edges(self.G, pos,
                                edge_color='gray',
                                arrows=True,
                                arrowsize=20,
                                width=2,
                                connectionstyle="arc3,rad=0.2")
            
            # Create labels with wrapped text
            labels = {}
            for node, data in self.G.nodes(data=True):
                if 'label' in data:
                    wrapped_text = textwrap.fill(data['label'], width=30)
                    labels[node] = wrapped_text
            
            nx.draw_networkx_labels(self.G, pos,
                                labels=labels,
                                font_size=6,
                                font_weight='bold')
            
            plt.title("Reasoning Tree Analysis\nGreen: Best Path, Blue: Alternative Thoughts",
                    fontsize=14,
                    pad=20)
            
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='lightgreen',
                                        markersize=15,
                                        label='Best Path'),
                            plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='lightblue',
                                        markersize=15,
                                        label='Alternative Thoughts')]
            plt.legend(handles=legend_elements,
                    loc='upper right',
                    fontsize=10)
            
            plt.tight_layout()
            
            if output_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path,
                        bbox_inches='tight',
                        dpi=300,
                        format='png')
                
                # Verify file was actually saved
                if os.path.exists(output_path):
                    success = True
                else:
                    print(f"Error: Failed to save visualization to {output_path}")
            else:
                success = True  # If no output path was specified, consider it successful
            
            return success
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
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
        self.best_path_ids = set()  # Track IDs of nodes in best path
        
    def _create_reasoning_prompt(self, context: str, current_path: List[ThoughtNode]) -> str:
        path_summary = "\n".join([
            f"- Analyzing {node.aspect}: {node.content}" 
            for node in current_path
        ])
        
        return f"""Analyze this research paper's publishability through careful reasoning.

Previous analysis steps:
{path_summary if current_path else "No previous analysis"}

Paper excerpt:
{context}

Based on the previous analysis (if any), YOUR TASK is TO PROVIDE {self.max_branches} DISTINCT reasoning thoughts about different aspects of the paper's publishability. Each thought should be thorough, well-supported, and explore a new dimension or deepen previous analysis.

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

    def generate_thoughts(self, context: str, current_path: List[ThoughtNode]) -> List[ThoughtNode]:
        prompt = self._create_reasoning_prompt(context, current_path)
        response = self.reasoning_llm.invoke(prompt)
        
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
                
            # Ensure proper tracking
            self.all_thoughts[node.node_id] = node
            self.visualizer.add_node(node, parent_node.node_id if parent_node else None)
            thoughts.append(node)
        
        return thoughts
    
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
    
    def evaluate_level(self, paths: List[List[ThoughtNode]], paper_content: str) -> Optional[PathEvaluation]:
        """Evaluate all paths at current depth and update their evaluations."""
        if not paths:
            return None
        
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                prompt = self._create_critic_prompt(paths, paper_content)
                evaluation = self.critic_llm.invoke(prompt)
                
                # Validate evaluation response
                if not evaluation or not isinstance(evaluation, PathEvaluation):
                    print("Warning: Invalid evaluation response from critic LLM")
                    retries += 1
                    if retries < max_retries:
                        print(f"Retrying evaluation (attempt {retries + 1}/{max_retries})...")
                        continue
                    raise ValueError("Failed to get valid evaluation from critic LLM after maximum retries")
                
                # Validate required fields
                if not evaluation.best_path or not evaluation.pruned_paths:
                    print("Warning: Missing required fields in evaluation")
                    retries += 1
                    if retries < max_retries:
                        print(f"Retrying evaluation (attempt {retries + 1}/{max_retries})...")
                        continue
                    raise ValueError("Failed to get complete evaluation fields after maximum retries")
                
                # Update node evaluations with categories
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
                raise  # Re-raise the exception after max retries



class PaperEvaluator:
    def __init__(self, reasoning_config: ModelConfig, critic_config: ModelConfig):
        reasoning_llm = LLMFactory.create_llm(reasoning_config)
        critic_llm = LLMFactory.create_llm(critic_config)
        
        self.tot = TreeOfThoughts(reasoning_llm, critic_llm)
        self.decision_llm = critic_llm.with_structured_output(PublishabilityDecision)
        
    def evaluate_paper(self, content: str) -> PublishabilityDecision:
        root_node = ThoughtNode(content="root", aspect="initial")
        paths_by_depth: Dict[int, List[List[ThoughtNode]]] = {0: [[root_node]]}  # Start with root
        valid_nodes: Set[str] = {root_node.node_id}
        
        best_path = None
        final_evaluation = None
        
        for depth in range(self.tot.max_depth):
            current_level_paths = []
            
            for path in paths_by_depth[depth]:
                if path[-1].node_id in valid_nodes:
                    thoughts = self.tot.generate_thoughts(content, path)
                    for thought in thoughts:
                        new_path = path + [thought]
                        current_level_paths.append(new_path)
            
            if not current_level_paths:
                break
                    
            evaluation = self.tot.evaluate_level(current_level_paths, content)
            
            if evaluation:
                valid_nodes = {evaluation.best_path} | set(evaluation.neutral_paths)
                paths_by_depth[depth + 1] = [p for p in current_level_paths 
                                        if p[-1].node_id in valid_nodes]
                
                # Track best path
                best_path = next(p for p in current_level_paths 
                            if p[-1].node_id == evaluation.best_path)
                final_evaluation = evaluation
        
        if best_path and final_evaluation:
            self.tot.mark_best_path(best_path)
            return self._make_final_decision(best_path, final_evaluation)
        
        raise ValueError("Failed to complete evaluation")
    
    def _make_final_decision(
        self,
        path: List[ThoughtNode],
        evaluation: PathEvaluation
    ) -> PublishabilityDecision:
        """Make final publishability decision based on best path."""
        path_content = "\n".join([
            f"Analysis of {node.aspect}:\n{node.content}\n"
            for node in path if node.content != "root"  # Skip root node
        ])
        
        prompt = f"""Based on this complete analysis path and its evaluation:

    Reasoning Path:
    {path_content}

    Make a final decision about the paper's publishability. Consider all aspects analyzed
    and provide concrete recommendations for improvement or acceptance.
    """
        final_decision = self.decision_llm.invoke(prompt)

        print("=======")
        print(f"Critic's evaluation rationale: {evaluation.rationale}")
        print("=======")
        print(f"Final Decision Response: {final_decision}")

        return final_decision


def main():
    # Configure models

    reasoning_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini"
    )
    
    critic_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o"
    )

    print("Configuring evaluator...")
    
    evaluator = PaperEvaluator(reasoning_config, critic_config)
    pdf_directory = "/home/divyansh/code/kdsh/dataset/Reference/Non-Publishable"
    pdf_path = "/home/divyansh/code/kdsh/dataset/Reference/Non-Publishable/R005.pdf"

    # Create output directory for visualizations and data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"paper_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(pdf_directory):
        print(f"Analyzing papers in {pdf_directory}...")
    else:
        print(f"Directory {pdf_directory} does not exist.")
        sys.exit(1)

    # Extract and analyze single paper
    print("Extracting PDF content...")
    content = extract_pdf_content(pdf_path)

    print("Content extracted from PDF")
    print("Analyzing paper...")

    # Get decision and visualization
    decision = evaluator.evaluate_paper(content)

    # Save tree visualization
    viz_path = os.path.join(output_dir, "thought_tree.png")
    if evaluator.tot.visualizer.visualize(viz_path):
        print(f"\nThought tree visualization saved to: {viz_path}")
    else:
        print("\nFailed to save thought tree visualization")

    # Save tree data as JSON
    tree_data = {
        "metadata": {
            "paper_path": pdf_path,
            "timestamp": timestamp,
            "analysis_result": decision.is_publishable
        },
        "nodes": [
            {
                "id": node_id,
                "content": node.content,
                "aspect": node.aspect,
                "evaluation": node.evaluation,
                "parent": node.parent_id
            }
            for node_id, node in evaluator.tot.all_thoughts.items()
        ],
        # Remove reference to decision.rationale and use recommendation instead
        "final_decision": {
            "is_publishable": decision.is_publishable,
            "primary_strengths": decision.primary_strengths,
            "critical_weaknesses": decision.critical_weaknesses,
            "recommendation": decision.recommendation
        }
    }

    json_path = os.path.join(output_dir, "thought_tree_data.json")
    with open(json_path, 'w') as f:
        json.dump(tree_data, f, indent=2)

    print(f"Thought tree data saved to: {json_path}")

    # Print analysis results
    print(f"\nAnalysis Results:")
    print(f"Publishable: {decision.is_publishable}")
    print("\nKey Strengths:")
    for strength in decision.primary_strengths:
        print(f"- {strength}")
    print("\nCritical Weaknesses:")
    for weakness in decision.critical_weaknesses:
        print(f"- {weakness}")
    print(f"\nRecommendation:\n{decision.recommendation}")

    # Create a summary file
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Paper Analysis Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Paper: {pdf_path}\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Publishable: {decision.is_publishable}\n\n")
        f.write("Key Strengths:\n")
        for strength in decision.primary_strengths:
            f.write(f"- {strength}\n")
        f.write("\nCritical Weaknesses:\n")
        for weakness in decision.critical_weaknesses:
            f.write(f"- {weakness}\n")
        f.write(f"\nRecommendation:\n{decision.recommendation}\n")
    
    print(f"\nAnalysis summary saved to: {summary_path}")


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

def process_papers_directory(base_dir: str, evaluator: PaperEvaluator) -> pd.DataFrame:
    """
    Process all papers in the given directory structure and create evaluation results.
    """
    results = []
    
    # Process Non-Publishable papers
    non_pub_dir = os.path.join(base_dir, "Non-Publishable")
    if os.path.exists(non_pub_dir):
        for filename in os.listdir(non_pub_dir):
            if filename.endswith('.pdf'):
                paper_path = os.path.join(non_pub_dir, filename)
                result = evaluate_single_paper(paper_path, False, evaluator)
                results.append(result)
                print(f"Completed processing: {filename}")
    
    # Process Publishable papers
    pub_dir = os.path.join(base_dir, "Publishable")
    if os.path.exists(pub_dir):
        for conf_dir in os.listdir(pub_dir):
            conf_path = os.path.join(pub_dir, conf_dir)
            if os.path.isdir(conf_path):
                for filename in os.listdir(conf_path):
                    if filename.endswith('.pdf'):
                        paper_path = os.path.join(conf_path, filename)
                        result = evaluate_single_paper(paper_path, True, evaluator)
                        results.append(result)
                        print(f"Completed processing: {filename}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = os.path.join(base_dir, "evaluation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df

def evaluate_single_paper(
    paper_path: str,
    true_label: bool,
    evaluator: PaperEvaluator
) -> Dict[str, Any]:
    """
    Extract content from a PDF and evaluate it.
    """
    print(f"\nProcessing: {paper_path}")
    
    try:
        # Step 1: Extract content from PDF
        print("Extracting PDF content...")
        content = extract_pdf_content(paper_path)
        
        # Step 2: Evaluate the content
        print("Evaluating content...")
        decision = evaluator.evaluate_paper(content)

        if check_content(content):
            pass

        
        # Create result dictionary
        result = {
            'paper_id': os.path.basename(paper_path),
            'true_label': true_label,
            'predicted_label': decision.is_publishable,
            'conference': os.path.basename(os.path.dirname(paper_path)) if true_label else 'Non-Publishable',
            'primary_strengths': '|'.join(decision.primary_strengths),
            'critical_weaknesses': '|'.join(decision.critical_weaknesses),
            'recommendation': decision.recommendation,
            'correct_prediction': true_label == decision.is_publishable,
            'file_path': paper_path
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {paper_path}: {str(e)}")
        return {
            'paper_id': os.path.basename(paper_path),
            'true_label': true_label,
            'predicted_label': None,
            'conference': os.path.basename(os.path.dirname(paper_path)) if true_label else 'Non-Publishable',
            'primary_strengths': '',
            'critical_weaknesses': '',
            'recommendation': f'ERROR: {str(e)}',
            'correct_prediction': False,
            'file_path': paper_path
        }

def analyze_results(df: pd.DataFrame) -> None:
    """
    Analyze and print evaluation results.
    """
    total_papers = len(df)
    valid_predictions = df['predicted_label'].notna()
    df_valid = df[valid_predictions]
    
    if len(df_valid) == 0:
        print("\nNo valid predictions to analyze!")
        return
        
    correct_predictions = df_valid['correct_prediction'].sum()
    accuracy = (correct_predictions / len(df_valid)) * 100
    
    print("\nEvaluation Results Analysis:")
    print(f"Total papers processed: {total_papers}")
    print(f"Papers with valid predictions: {len(df_valid)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Confusion matrix for valid predictions
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

def test_reference():
    # Configure models
    reasoning_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini"
    )
    
    critic_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o"
    )

    print("Configuring evaluator...")
    evaluator = PaperEvaluator(reasoning_config, critic_config)
    
    # Base directory containing the papers
    base_dir = "/home/divyansh/code/kdsh/dataset/Reference"
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_base_dir = f"paper_analyses_{timestamp}"
    os.makedirs(analysis_base_dir, exist_ok=True)
    
    results = []
    
    # Process Non-Publishable papers
    non_pub_dir = os.path.join(base_dir, "Non-Publishable")
    if os.path.exists(non_pub_dir):
        for filename in os.listdir(non_pub_dir):
            if filename.endswith('.pdf'):
                paper_path = os.path.join(non_pub_dir, filename)
                result = process_single_paper(paper_path, False, evaluator, analysis_base_dir)
                results.append(result)
                print(f"Completed processing: {filename}")
    
    # Process Publishable papers
    pub_dir = os.path.join(base_dir, "Publishable")
    if os.path.exists(pub_dir):
        for conf_dir in os.listdir(pub_dir):
            conf_path = os.path.join(pub_dir, conf_dir)
            if os.path.isdir(conf_path):
                for filename in os.listdir(conf_path):
                    if filename.endswith('.pdf'):
                        paper_path = os.path.join(conf_path, filename)
                        result = process_single_paper(paper_path, True, evaluator, analysis_base_dir)
                        results.append(result)
                        print(f"Completed processing: {filename}")
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    csv_path = os.path.join(analysis_base_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Analyze and print results
    analyze_results(df)

def process_single_paper(
    paper_path: str,
    true_label: bool,
    evaluator: PaperEvaluator,
    analysis_base_dir: str
) -> Dict[str, Any]:
    """
    Process a single paper and generate all required outputs.
    """
    print(f"\nProcessing: {paper_path}")
    
    try:
        # Create paper-specific output directory
        paper_id = os.path.basename(paper_path).replace('.pdf', '')
        paper_output_dir = os.path.join(analysis_base_dir, paper_id)
        os.makedirs(paper_output_dir, exist_ok=True)
        
        # Extract content from PDF
        print("Extracting PDF content...")
        content = extract_pdf_content(paper_path)
        
        # Evaluate the content
        print("Evaluating content...")
        decision = evaluator.evaluate_paper(content)
        
        # Save tree visualization
        viz_path = os.path.join(paper_output_dir, f"{paper_id}_thought_tree.png")
        if evaluator.tot.visualizer.visualize(viz_path):
            print(f"Thought tree visualization saved to: {viz_path}")
        else:
            print("Failed to save thought tree visualization")
        
        # Save tree data as JSON
        tree_data = {
            "metadata": {
                "paper_path": paper_path,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "analysis_result": decision.is_publishable
            },
            "nodes": [
                {
                    "id": node_id,
                    "content": node.content,
                    "aspect": node.aspect,
                    "evaluation": node.evaluation,
                    "parent": node.parent_id
                }
                for node_id, node in evaluator.tot.all_thoughts.items()
            ],
            "final_decision": {
                "is_publishable": decision.is_publishable,
                "primary_strengths": decision.primary_strengths,
                "critical_weaknesses": decision.critical_weaknesses,
                "recommendation": decision.recommendation
            }
        }
        
        json_path = os.path.join(paper_output_dir, f"{paper_id}_thought_tree_data.json")
        with open(json_path, 'w') as f:
            json.dump(tree_data, f, indent=2)
        
        # Create analysis summary
        summary_path = os.path.join(paper_output_dir, f"{paper_id}_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Paper Analysis Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Paper: {paper_path}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Publishable: {decision.is_publishable}\n\n")
            f.write("Key Strengths:\n")
            for strength in decision.primary_strengths:
                f.write(f"- {strength}\n")
            f.write("\nCritical Weaknesses:\n")
            for weakness in decision.critical_weaknesses:
                f.write(f"- {weakness}\n")
            f.write(f"\nRecommendation:\n{decision.recommendation}\n")
        
        # Create result dictionary for CSV
        result = {
            'paper_id': paper_id,
            'true_label': true_label,
            'predicted_label': decision.is_publishable,
            'conference': os.path.basename(os.path.dirname(paper_path)) if true_label else 'Non-Publishable',
            'primary_strengths': '|'.join(decision.primary_strengths),
            'critical_weaknesses': '|'.join(decision.critical_weaknesses),
            'recommendation': decision.recommendation,
            'correct_prediction': true_label == decision.is_publishable,
            'file_path': paper_path,
            'output_dir': paper_output_dir
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {paper_path}: {str(e)}")
        return {
            'paper_id': os.path.basename(paper_path).replace('.pdf', ''),
            'true_label': true_label,
            'predicted_label': None,
            'conference': os.path.basename(os.path.dirname(paper_path)) if true_label else 'Non-Publishable',
            'primary_strengths': '',
            'critical_weaknesses': '',
            'recommendation': f'ERROR: {str(e)}',
            'correct_prediction': False,
            'file_path': paper_path,
            'output_dir': ''
        }

def test_papers():
    # Configure models
    reasoning_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini"
    )
    
    critic_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o"
    )

    print("Configuring evaluator...")
    evaluator = PaperEvaluator(reasoning_config, critic_config)
    
    # Base directory containing the papers
    base_dir = "/home/divyansh/code/kdsh/dataset"
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_base_dir = f"paper_analyses_{timestamp}"
    os.makedirs(analysis_base_dir, exist_ok=True)
    
    results = []
    
    paper_dir = os.path.join(base_dir, "Papers")
    if os.path.exists(paper_dir):
        filenames = sorted([f for f in os.listdir(paper_dir) if f.endswith('.pdf')])
        
        for filename in filenames[:50]:  # Process the first 50 papers sequentially
            paper_path = os.path.join(paper_dir, filename)
            result = process_single_paper(paper_path, False, evaluator, analysis_base_dir)
            results.append(result)
            print(f"Completed processing: {filename}")
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    csv_path = os.path.join(analysis_base_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Analyze and print results
    analyze_results(df)

def check_content(content):
    detector = checker()
    print(f"Checking content of {content[:50]}...")
    results = detector._run(content)
    return results

if __name__ == "__main__":
    # main()
    # test_reference()
    # test_papers()
    # reasoning_config = ModelConfig(
    #     provider=ModelProvider.OPENAI,
    #     model_name="gpt-4o-mini"
    # )

    content_r007 = """
Advancements in 3D Food Modeling: A Review of the MetaFood Challenge Techniques and Outcomes

Abstract

The growing focus on leveraging computer vision for dietary oversight and nutri- tion tracking has spurred the creation of sophisticated 3D reconstruction methods for food. The lack of comprehensive, high-fidelity data, coupled with limited collaborative efforts between academic and industrial sectors, has significantly hindered advancements in this domain. This study addresses these obstacles by introducing the MetaFood Challenge, aimed at generating precise, volumetrically accurate 3D food models from 2D images, utilizing a checkerboard for size cal- ibration. The challenge was structured around 20 food items across three levels of complexity: easy (200 images), medium (30 images), and hard (1 image). A total of 16 teams participated in the final assessment phase. The methodologies developed during this challenge have yielded highly encouraging outcomes in 3D food reconstruction, showing great promise for refining portion estimation in dietary evaluations and nutritional tracking. Further information on this workshop challenge and the dataset is accessible via the provided URL.1 Introduction

The convergence of computer vision technologies with culinary practices has pioneered innovative approaches to dietary monitoring and nutritional assessment. The MetaFood Workshop Challenge represents a landmark initiative in this emerging field, responding to the pressing demand for precise and scalable techniques for estimating food portions and monitoring nutritional consumption. Such technologies are vital for fostering healthier eating behaviors and addressing health issues linked to diet.

By concentrating on the development of accurate 3D models of food derived from various visual inputs, including multiple views and single perspectives, this challenge endeavors to bridge the disparity between current methodologies and practical needs. It promotes the creation of unique solutions capable of managing the intricacies of food morphology, texture, and illumination, while also meeting the real-world demands of dietary evaluation. This initiative gathers experts from computer vision, machine learning, and nutrition science to propel 3D food reconstruction technologies forward. These advancements have the potential to substantially enhance the precision and utility of food portion estimation across diverse applications, from individual health tracking to extensive nutritional investigations.

Conventional methods for assessing diet, like 24-Hour Recall or Food Frequency Questionnaires (FFQs), are frequently reliant on manual data entry, which is prone to inaccuracies and can be burdensome. The lack of 3D data in 2D RGB food images further complicates the use of regression- based methods for estimating food portions directly from images of eating occasions. By enhancing 3D reconstruction for food, the aim is to provide more accurate and intuitive nutritional assessment tools. This technology could revolutionize the sharing of culinary experiences and significantly impact nutrition science and public health.Participants were tasked with creating 3D models of 20 distinct food items from 2D images, mim- icking scenarios where mobile devices equipped with depth-sensing cameras are used for dietary

recording and nutritional tracking. The challenge was segmented into three tiers of difficulty based on the number of images provided: approximately 200 images for easy, 30 for medium, and a single top-view image for hard. This design aimed to rigorously test the adaptability and resilience of proposed solutions under various realistic conditions. A notable feature of this challenge was the use of a visible checkerboard for physical referencing and the provision of depth images for each frame, ensuring the 3D models maintained accurate real-world measurements for portion size estimation.

This initiative not only expands the frontiers of 3D reconstruction technology but also sets the stage for more reliable and user-friendly real-world applications, including image-based dietary assessment. The resulting solutions hold the potential to profoundly influence nutritional intake monitoring and comprehension, supporting broader health and wellness objectives. As progress continues, innovative applications are anticipated to transform personal health management, nutritional research, and the wider food industry. The remainder of this report is structured as follows: Section 2 delves into the existing literature on food portion size estimation, Section 3 describes the dataset and evaluation framework used in the challenge, and Sections 4, 5, and 6 discuss the methodologies and findings of the top three teams (VoIETA, ININ-VIAUN, and FoodRiddle), respectively.2 Related Work

Estimating food portions is a crucial part of image-based dietary assessment, aiming to determine the volume, energy content, or macronutrients directly from images of meals. Unlike the well-studied task of food recognition, estimating food portions is particularly challenging due to the lack of 3D information and physical size references necessary for accurately judging the actual size of food portions. Accurate portion size estimation requires understanding the volume and density of food, elements that are hard to deduce from a 2D image, underscoring the need for sophisticated techniques to tackle this problem. Current methods for estimating food portions are grouped into four categories.

Stereo-Based Approaches use multiple images to reconstruct the 3D structure of food. Some methods estimate food volume using multi-view stereo reconstruction based on epipolar geometry, while others perform two-view dense reconstruction. Simultaneous Localization and Mapping (SLAM) has also been used for continuous, real-time food volume estimation. However, these methods are limited by their need for multiple images, which is not always practical.

Model-Based Approaches use predefined shapes and templates to estimate volume. For instance, certain templates are assigned to foods from a library and transformed based on physical references to estimate the size and location of the food. Template matching approaches estimate food volume from a single image, but they struggle with variations in food shapes that differ from predefined templates. Recent work has used 3D food meshes as templates to align camera and object poses for portion size estimation.Depth Camera-Based Approaches use depth cameras to create depth maps, capturing the distance from the camera to the food. These depth maps form a voxel representation used for volume estimation. The main drawback is the need for high-quality depth maps and the extra processing required for consumer-grade depth sensors.

Deep Learning Approaches utilize neural networks trained on large image datasets for portion estimation. Regression networks estimate the energy value of food from single images or from an "Energy Distribution Map" that maps input images to energy distributions. Some networks use both images and depth maps to estimate energy, mass, and macronutrient content. However, deep learning methods require extensive data for training and are not always interpretable, with performance degrading when test images significantly differ from training data.

While these methods have advanced food portion estimation, they face limitations that hinder their widespread use and accuracy. Stereo-based methods are impractical for single images, model-based approaches struggle with diverse food shapes, depth camera methods need specialized hardware, and deep learning approaches lack interpretability and struggle with out-of-distribution samples. 3D reconstruction offers a promising solution by providing comprehensive spatial information, adapting to various shapes, potentially working with single images, offering visually interpretable results, and enabling a standardized approach to food portion estimation. These benefits motivated the organization of the 3D Food Reconstruction challenge, aiming to overcome existing limitations and

develop more accurate, user-friendly, and widely applicable food portion estimation techniques, impacting nutritional assessment and dietary monitoring.

3 Datasets and Evaluation Pipeline3.1 Dataset Description

The dataset for the MetaFood Challenge features 20 carefully chosen food items from the MetaFood3D dataset, each scanned in 3D and accompanied by video recordings. To ensure precise size accuracy in the reconstructed 3D models, each food item was captured alongside a checkerboard and pattern mat, serving as physical scaling references. The challenge is divided into three levels of difficulty, determined by the quantity of 2D images provided for reconstruction:

¢ Easy: Around 200 images taken from video.

* Medium: 30 images.

¢ Hard: A single image from a top-down perspective.

Table 1 details the food items included in the dataset.

Table 1: MetaFood Challenge Data Details

Object Index Food Item Difficulty Level Number of Frames 1 Strawberry Easy 199 2 Cinnamon bun Easy 200 3 Pork rib Easy 200 4 Corn Easy 200 5 French toast Easy 200 6 Sandwich Easy 200 7 Burger Easy 200 8 Cake Easy 200 9 Blueberry muffin Medium 30 10 Banana Medium 30 11 Salmon Medium 30 12 Steak Medium 30 13 Burrito Medium 30 14 Hotdog Medium 30 15 Chicken nugget Medium 30 16 Everything bagel Hard 1 17 Croissant Hard 1 18 Shrimp Hard 1 19 Waffle Hard 1 20 Pizza Hard 1

3.2 Evaluation Pipeline

The evaluation process is split into two phases, focusing on the accuracy of the reconstructed 3D models in terms of shape (3D structure) and portion size (volume).

3.2.1 Phase-I: Volume Accuracy

In the first phase, the Mean Absolute Percentage Error (MAPE) is used to evaluate portion size accuracy, calculated as follows:

12 MAPE = — > i=l Ai — Fi Aj x 100% qd)

where A; is the actual volume (in ml) of the i-th food item obtained from the scanned 3D food mesh, and F; is the volume calculated from the reconstructed 3D mesh.3.2.2 Phase-II: Shape Accuracy

Teams that perform well in Phase-I are asked to submit complete 3D mesh files for each food item. This phase involves several steps to ensure precision and fairness:

* Model Verification: Submitted models are checked against the final Phase-I submissions for consistency, and visual inspections are conducted to prevent rule violations.

* Model Alignment: Participants receive ground truth 3D models and a script to compute the final Chamfer distance. They must align their models with the ground truth and prepare a transformation matrix for each submitted object. The final Chamfer distance is calculated using these models and matrices.

¢ Chamfer Distance Calculation: Shape accuracy is assessed using the Chamfer distance metric. Given two point sets X and Y,, the Chamfer distance is defined as:

4 > 1 2 dev(X.Y) = 15 Do mip lle — yll2 + Ty DL main lle — all (2) EX yey

This metric offers a comprehensive measure of similarity between the reconstructed 3D models and the ground truth. The final ranking is determined by combining scores from both Phase-I (volume accuracy) and Phase-II (shape accuracy). Note that after the Phase-I evaluation, quality issues were found with the data for object 12 (steak) and object 15 (chicken nugget), so these items were excluded from the final overall evaluation.

4 First Place Team - VoIETA

4.1 Methodology

The team’s research employs multi-view reconstruction to generate detailed food meshes and calculate precise food volumes.4.1.1 Overview

The team’s method integrates computer vision and deep learning to accurately estimate food volume from RGBD images and masks. Keyframe selection ensures data quality, supported by perceptual hashing and blur detection. Camera pose estimation and object segmentation pave the way for neural surface reconstruction, creating detailed meshes for volume estimation. Refinement steps, including isolated piece removal and scaling factor adjustments, enhance accuracy. This approach provides a thorough solution for accurate food volume assessment, with potential uses in nutrition analysis.4.1.2 The Team’s Proposal: VoIETA

The team starts by acquiring input data, specifically RGBD images and corresponding food object masks. The RGBD images, denoted as Ip = {Ip;}"_,, where n is the total number of frames, provide depth information alongside RGB images. The food object masks, {uf }"_,, help identify regions of interest within these images.

Next, the team selects keyframes. From the set {Ip;}7_1, keyframes {If }4_, C {Ipi}f_4 are chosen. A method is implemented to detect and remove duplicate and blurry images, ensuring high-quality frames. This involves applying a Gaussian blurring kernel followed by the fast Fourier transform method. Near-Image Similarity uses perceptual hashing and Hamming distance threshold- ing to detect similar images and retain overlapping ones. Duplicates and blurry images are excluded to maintain data integrity and accuracy.

Using the selected keyframes {I if }*_ |, the team estimates camera poses through a method called PixSfM, which involves extracting features using SuperPoint, matching them with SuperGlue, and refining them. The outputs are the camera poses {Cj} Ro crucial for understanding the scene’s spatial layout.

In parallel, the team uses a tool called SAM for reference object segmentation. SAM segments the reference object with a user-provided prompt, producing a reference object mask /" for each keyframe. This mask helps track the reference object across all frames. The XMem++ method extends the reference object mask /¥ to all frames, creating a comprehensive set of reference object masks {/?}"_,. This ensures consistent reference object identification throughout the dataset.

To create RGBA images, the team combines RGB images, reference object masks {M/??}"_,, and food object masks {/}"}"_,. This step, denoted as {J}? }"~,, integrates various data sources into a unified format for further processing.The team converts the RGBA images {I/*}?_, and camera poses {C}}4_, into meaningful metadata and modeled data D,,,. This transformation facilitates accurate scene reconstruction.

The modeled data D,,, is input into NeuS2 for mesh reconstruction. NeuS2 generates colorful meshes {R/, R"} for the reference and food objects, providing detailed 3D representations. The team uses the "Remove Isolated Pieces" technique to refine the meshes. Given that the scenes contain only one food item, the diameter threshold is set to 5% of the mesh size. This method deletes isolated connected components with diameters less than or equal to 5%, resulting in a cleaned mesh {RC , RC’}. This step ensures that only significant parts of the mesh are retained.

The team manually identifies an initial scaling factor S using the reference mesh via MeshLab. This factor is fine-tuned to Sy using depth information and food and reference masks, ensuring accurate scaling relative to real-world dimensions. Finally, the fine-tuned scaling factor S, is applied to the cleaned food mesh RCS, producing the final scaled food mesh RF. This step culminates in an accurately scaled 3D representation of the food object, enabling precise volume estimation.4.1.3 Detecting the scaling factor

Generally, 3D reconstruction methods produce unitless meshes by default. To address this, the team manually determines the scaling factor by measuring the distance for each block of the reference object mesh. The average of all block lengths [ay is calculated, while the actual real-world length is constant at ca; = 0.012 meters. The scaling factor S = lpeat /lavg is applied to the clean food mesh RC! , resulting in the final scaled food mesh RFS in meters.

The team uses depth information along with food and reference object masks to validate the scaling factors. The method for assessing food size involves using overhead RGB images for each scene. Initially, the pixel-per-unit (PPU) ratio (in meters) is determined using the reference object. Subse- quently, the food width (f,,,) and length (f7) are extracted using a food object mask. To determine the food height (f;,), a two-step process is followed. First, binary image segmentation is performed using the overhead depth and reference images, yielding a segmented depth image for the reference object. The average depth is then calculated using the segmented reference object depth (d,-). Similarly, employing binary image segmentation with an overhead food object mask and depth image, the average depth for the segmented food depth image (d+) is computed. The estimated food height f), is the absolute difference between d, and dy. To assess the accuracy of the scaling factor S, the food bounding box volume (f,, x fi x fn) x PPU is computed. The team evaluates if the scaling factor S' generates a food volume close to this potential volume, resulting in S'sjn¢. Table 2 lists the scaling factors, PPU, 2D reference object dimensions, 3D food object dimensions, and potential volume.For one-shot 3D reconstruction, the team uses One-2-3-45 to reconstruct a 3D model from a single RGBA view input after applying binary image segmentation to both food RGB and mask images. Isolated pieces are removed from the generated mesh, and the scaling factor S', which is closer to the potential volume of the clean mesh, is reused.

4.2 Experimental Results

4.2.1 Implementation settings

Experiments were conducted using two GPUs: GeForce GTX 1080 Ti/12G and RTX 3060/6G. The Hamming distance for near image similarity was set to 12. For Gaussian kernel radius, even numbers in the range [0...30] were used for detecting blurry images. The diameter for removing isolated pieces was set to 5%. NeuS2 was run for 15,000 iterations with a mesh resolution of 512x512, a unit cube "aabb scale" of 1, "scale" of 0.15, and "offset" of [0.5, 0.5, 0.5] for each food scene.4.2.2 VolETA Results

The team extensively validated their approach on the challenge dataset and compared their results with ground truth meshes using MAPE and Chamfer distance metrics. The team’s approach was applied separately to each food scene. A one-shot food volume estimation approach was used if the number of keyframes k equaled 1; otherwise, a few-shot food volume estimation was applied. Notably, the keyframe selection process chose 34.8% of the total frames for the rest of the pipeline, showing the minimum frames with the highest information.

Table 2: List of Extracted Information Using RGBD and Masks

Level Id Label Sy PPU Ry x Ri (fw x fi x fh) 1 Strawberry 0.08955223881 0.01786 320 x 360 = (238 x 257 x 2.353) 2 Cinnamon bun 0.1043478261 0.02347 236 x 274 = (363 x 419 x 2.353) 3 Pork rib 0.1043478261 0.02381 246x270 (435 x 778 x 1.176) Easy 4 Corn 0.08823529412 0.01897 291 x 339 (262 x 976 x 2.353) 5 French toast 0.1034482759 0.02202 266 x 292 (530 x 581 x 2.53) 6 Sandwich 0.1276595745 0.02426 230 x 265 (294 x 431 x 2.353) 7 Burger 0.1043478261 0.02435 208 x 264 (378 x 400 x 2.353) 8 Cake 0.1276595745 0.02143. 256 x 300 = (298 x 310 x 4.706) 9 Blueberry muffin —_0.08759124088 0.01801 291x357 (441 x 443 x 2.353) 10 Banana 0.08759124088 0.01705 315x377 (446 x 857 x 1.176) Medium 11 Salmon 0.1043478261 0.02390 242 x 269 (201 x 303 x 1.176) 13 Burrito 0.1034482759 0.02372 244 x 27 (251 x 917 x 2.353) 14 Frankfurt sandwich —_0.1034482759 0.02115. 266 x 304. (400 x 1022 x 2.353) 16 Everything bagel —_0.08759124088 0.01747 306 x 368 = (458 x 134 x 1.176 ) ) Hard 17 Croissant 0.1276595745 0.01751 319 x 367 = (395 x 695 x 2.176 18 Shrimp 0.08759124088 0.02021 249x318 (186 x 95 x 0.987) 19 Waffle 0.01034482759 0.01902 294 x 338 (465 x 537 x 0.8) 20 Pizza 0.01034482759 0.01913 292 x 336 (442 x 651 x 1.176)After finding keyframes, PixSfM estimated the poses and point cloud. After generating scaled meshes, the team calculated volumes and Chamfer distance with and without transformation metrics. Meshes were registered with ground truth meshes using ICP to obtain transformation metrics.

Table 3 presents quantitative comparisons of the team’s volumes and Chamfer distance with and without estimated transformation metrics from ICP. For overall method performance, Table 4 shows the MAPE and Chamfer distance with and without transformation metrics.

Additionally, qualitative results on one- and few-shot 3D reconstruction from the challenge dataset are shown. The model excels in texture details, artifact correction, missing data handling, and color adjustment across different scene parts.

Limitations: Despite promising results, several limitations need to be addressed in future work:

¢ Manual processes: The current pipeline includes manual steps like providing segmentation prompts and identifying scaling factors, which should be automated to enhance efficiency.

¢ Input requirements: The method requires extensive input information, including food masks and depth data. Streamlining these inputs would simplify the process and increase applicability.

* Complex backgrounds and objects: The method has not been tested in environments with complex backgrounds or highly intricate food objects.

¢ Capturing complexities: The method has not been evaluated under different capturing complexities, such as varying distances and camera speeds.

¢ Pipeline complexity: For one-shot neural rendering, the team currently uses One-2-3-45. They aim to use only the 2D diffusion model, Zero123, to reduce complexity and improve efficiency.

Table 3: Quantitative Comparison with Ground Truth Using Chamfer DistanceL Id Team’s Vol. GT Vol. Ch. w/tm Ch. w/otm 1 40.06 38.53 1.63 85.40 2 216.9 280.36 TA2 111.47 3 278.86 249.67 13.69 172.88 E 4 279.02 295.13 2.03 61.30 5 395.76 392.58 13.67 102.14 6 205.17 218.44 6.68 150.78 7 372.93 368.77 4.70 66.91 8 186.62 73.13 2.98 152.34 9 224.08 232.74 3.91 160.07 10 153.76 63.09 2.67 138.45 M ill 80.4 85.18 3.37 151.14 13 363.99 308.28 5.18 147.53 14 535.44 589.83 4.31 89.66 16 163.13 262.15 18.06 28.33 H 17 224.08 81.36 9.44 28.94 18 25.4 20.58 4.28 12.84 19 110.05 08.35 11.34 23.98 20 130.96 19.83 15.59 31.05

Table 4: Quantitative Comparison with Ground Truth Using MAPE and Chamfer Distance

MAPE Ch. w/t.m Ch. w/o t.m (%) sum mean sum mean 10.973 0.130 0.007 1.715 0.095

5 Second Place Team - ININ-VIAUN

5.1 Methodology

This section details the team’s proposed network, illustrating the step-by-step process from original images to final mesh models.5.1.1 Scale factor estimation

The procedure for estimating the scale factor at the coordinate level is illustrated in Figure 9. The team adheres to a method involving corner projection matching. Specifically, utilizing the COLMAP dense model, the team acquires the pose of each image along with dense point cloud data. For any given image img, and its extrinsic parameters [R|t];,, the team initially performs threshold-based corner detection, setting the threshold at 240. This step allows them to obtain the pixel coordinates of all detected corners. Subsequently, using the intrinsic parameters k and the extrinsic parameters [R|t],, the point cloud is projected onto the image plane. Based on the pixel coordinates of the corners, the team can identify the closest point coordinates P* for each corner, where i represents the index of the corner. Thus, they can calculate the distance between any two corners as follows:

Di =(PE- PPP Wid G 6)

To determine the final computed length of each checkerboard square in image k, the team takes the minimum value of each row of the matrix D” (excluding the diagonal) to form the vector d*. The median of this vector is then used. The final scale calculation formula is given by Equation 4, where 0.012 represents the known length of each square (1.2 cm):

0.012 scale = —,——__~ 4 eae Ss med) ”5.1.2 3D Reconstruction

The 3D reconstruction process, depicted in Figure 10, involves two different pipelines to accommodate variations in input viewpoints. The first fifteen objects are processed using one pipeline, while the last five single-view objects are processed using another.

For the initial fifteen objects, the team uses COLMAP to estimate poses and segment the food using the provided segment masks. Advanced multi-view 3D reconstruction methods are then applied to reconstruct the segmented food. The team employs three different reconstruction methods: COLMAP, DiffusioNeRF, and NeRF2Mesh. They select the best reconstruction results from these methods and extract the mesh. The extracted mesh is scaled using the estimated scale factor, and optimization techniques are applied to obtain a refined mesh.

For the last five single-view objects, the team experiments with several single-view reconstruction methods, including Zero123, Zerol23++, One2345, ZeroNVS, and DreamGaussian. They choose ZeroNVS to obtain a 3D food model consistent with the distribution of the input image. The intrinsic camera parameters from the fifteenth object are used, and an optimization method based on reprojection error refines the extrinsic parameters of the single camera. Due to limitations in single-view reconstruction, depth information from the dataset and the checkerboard in the monocular image are used to determine the size of the extracted mesh. Finally, optimization techniques are applied to obtain a refined mesh.5.1.3 Mesh refinement

During the 3D Reconstruction phase, it was observed that the model’s results often suffered from low quality due to holes on the object’s surface and substantial noise, as shown in Figure 11.

To address the holes, MeshFix, an optimization method based on computational geometry, is em- ployed. For surface noise, Laplacian Smoothing is used for mesh smoothing operations. The Laplacian Smoothing method adjusts the position of each vertex to the average of its neighboring vertices:

1 (new) __ y7(old) (old) (old) anes (Cee JEN (i)

In their implementation, the smoothing factor X is set to 0.2, and 10 iterations are performed.

5.2. Experimental Results

5.2.1 Estimated scale factor

The scale factors estimated using the described method are shown in Table 5. Each image and the corresponding reconstructed 3D model yield a scale factor, and the table presents the average scale factor for each object.

5.2.2 Reconstructed meshes

The refined meshes obtained using the described methods are shown in Figure 12. The predicted model volumes, ground truth model volumes, and the percentage errors between them are presented in Table 6.5.2.3. Alignment

The team designs a multi-stage alignment method for evaluating reconstruction quality. Figure 13 illustrates the alignment process for Object 14. First, the central points of both the predicted and ground truth models are calculated, and the predicted model is moved to align with the central point of the ground truth model. Next, ICP registration is performed for further alignment, significantly reducing the Chamfer distance. Finally, gradient descent is used for additional fine-tuning to obtain the final transformation matrix.

The total Chamfer distance between all 18 predicted models and the ground truths is 0.069441 169.

Table 5: Estimated Scale Factors

Object Index Food Item Scale Factor 1 Strawberry 0.060058 2 Cinnamon bun 0.081829 3 Pork rib 0.073861 4 Corn 0.083594 5 French toast 0.078632 6 Sandwich 0.088368 7 Burger 0.103124 8 Cake 0.068496 9 Blueberry muffin 0.059292 10 Banana 0.058236 11 Salmon 0.083821 13 Burrito 0.069663 14 Hotdog 0.073766

Table 6: Metric of Volume

Object Index Predicted Volume Ground Truth Error Percentage 1 44.51 38.53 15.52 2 321.26 280.36 14.59 3 336.11 249.67 34.62 4 347.54 295.13 17.76 5 389.28 392.58 0.84 6 197.82 218.44 9.44 7 412.52 368.77 11.86 8 181.21 173.13 4.67 9 233.79 232.74 0.45 10 160.06 163.09 1.86 11 86.0 85.18 0.96 13 334.7 308.28 8.57 14 517.75 589.83 12.22 16 176.24 262.15 32.77 17 180.68 181.36 0.37 18 13.58 20.58 34.01 19 117.72 108.35 8.64 20 117.43 119.83 20.03

6 Best 3D Mesh Reconstruction Team - FoodRiddle6.1 Methodology

To achieve high-fidelity food mesh reconstruction, the team developed two procedural pipelines as depicted in Figure 14. For simple and medium complexity cases, they employed a structure-from- motion strategy to ascertain the pose of each image, followed by mesh reconstruction. Subsequently, a sequence of post-processing steps was implemented to recalibrate the scale and improve mesh quality. For cases involving only a single image, the team utilized image generation techniques to facilitate model generation.

6.1.1 Multi- View Reconstruction

For Structure from Motion (SfM), the team enhanced the advanced COLMAP method by integrating SuperPoint and SuperGlue techniques. This integration significantly addressed the issue of limited keypoints in scenes with minimal texture, as illustrated in Figure 15.

In the mesh reconstruction phase, the team’s approach builds upon 2D Gaussian Splatting, which employs a differentiable 2D Gaussian renderer and includes regularization terms for depth distortion

and normal consistency. The Truncated Signed Distance Function (TSDF) results are utilized to produce a dense point cloud.

During post-processing, the team applied filtering and outlier removal methods, identified the outline of the supporting surface, and projected the lower mesh vertices onto this surface. They utilized the reconstructed checkerboard to correct the model’s scale and employed Poisson reconstruction to create a complete, watertight mesh of the subject.6.1.2 Single-View Reconstruction

For 3D reconstruction from a single image, the team utilized advanced methods such as LGM, Instant Mesh, and One-2-3-45 to generate an initial mesh. This initial mesh was then refined in conjunction with depth structure information.

To adjust the scale, the team estimated the object’s length using the checkerboard as a reference, assuming that the object and the checkerboard are on the same plane. They then projected the 3D object back onto the original 2D image to obtain a more precise scale for the object.

6.2. Experimental Results

Through a process of nonlinear optimization, the team sought to identify a transformation that minimizes the Chamfer distance between their mesh and the ground truth mesh. This optimization aimed to align the two meshes as closely as possible in three-dimensional space. Upon completion of this process, the average Chamfer dis- tance across the final reconstructions of the 20 objects amounted to 0.0032175 meters. As shown in Table 7, Team FoodRiddle achieved the best scores for both multi- view and single-view reconstructions, outperforming other teams in the competition.

Table 7: Total Errors for Different Teams on Multi-view and Single-view Data

Team Multi-view (1-14) Single-view (16-20) FoodRiddle 0.036362 0.019232 ININ-VIAUN 0.041552 0.027889 VolETA 0.071921 0.0587267 Conclusion

This report examines and compiles the techniques and findings from the MetaFood Workshop challenge on 3D Food Reconstruction. The challenge sought to enhance 3D reconstruction methods by concentrating on food items, tackling the distinct difficulties presented by varied textures, reflective surfaces, and intricate geometries common in culinary subjects.

The competition involved 20 diverse food items, captured under various conditions and with differing numbers of input images, specifically designed to challenge participants in creating robust reconstruc- tion models. The evaluation was based on a two-phase process, assessing both portion size accuracy through Mean Absolute Percentage Error (MAPE) and shape accuracy using the Chamfer distance metric.

Of all participating teams, three reached the final submission stage, presenting a range of innovative solutions. Team VolETA secured first place with the best overall performance in both Phase-I and Phase-II, followed by team ININ-VIAUN in second place. Additionally, the FoodRiddle team exhibited superior performance in Phase-II, highlighting a competitive and high-caliber field of entries for 3D mesh reconstruction. The challenge has successfully advanced the field of 3D food reconstruction, demonstrating the potential for accurate volume estimation and shape reconstruction in nutritional analysis and food presentation applications. The novel methods developed by the participating teams establish a strong foundation for future research in this area, potentially leading to more precise and user-friendly approaches for dietary assessment and monitoring.

10
"""
    content_r001 = """"
    Transdimensional Properties of Graphite in Relation to Cheese Consumption on Tuesday Afternoons

Abstract

Graphite research has led to discoveries about dolphins and their penchant for collecting rare flowers, which bloom only under the light of a full moon, while simultaneously revealing the secrets of dark matter and its relation to the perfect recipe for chicken parmesan, as evidenced by the curious case of the missing socks in the laundry basket, which somehow correlates with the migration patterns of but- terflies and the art of playing the harmonica underwater, where the sounds produced are eerily similar to the whispers of ancient forests, whispering tales of forgotten civilizations and their advanced understanding of quantum mechanics, applied to the manufacture of sentient toasters that can recite Shakespearean sonnets, all of which is connected to the inherent properties of graphite and its ability to conduct the thoughts of extraterrestrial beings, who are known to communicate through a complex system of interpretive dance and pastry baking, culminating in a profound understanding of the cosmos, as reflected in the intricate patterns found on the surface of a butterfly’s wings, and the uncanny resemblance these patterns bear to the molecular structure of graphite, which holds the key to unlocking the secrets of time travel and the optimal method for brewing coffee.

1 IntroductionThe fascinating realm of graphite has been juxtaposed with the intricacies of quantum mechanics, wherein the principles of superposition and entanglement have been observed to influence the baking of croissants, a phenomenon that warrants further investigation, particularly in the context of flaky pastry crusts, which, incidentally, have been found to exhibit a peculiar affinity for the sonnets of Shakespeare, specifically Sonnet 18, whose themes of beauty and mortality have been linked to the existential implications of graphitic carbon, a subject that has garnered significant attention in recent years, notwithstanding the fact that the aerodynamic properties of graphite have been studied extensively in relation to the flight patterns of migratory birds, such as the Arctic tern, which, intriguingly, has been known to incorporate graphite particles into its nest-building materials, thereby potentially altering the structural integrity of the nests, a consideration that has led researchers to explore the role of graphite in the development of more efficient wind turbine blades, an application that has been hindered by the limitations of current manufacturing techniques, which, paradoxically, have been inspired by the ancient art of Egyptian hieroglyphics, whose symbolic representations of graphite have been interpreted as a harbinger of good fortune, a notion that has been debunked by scholars of ancient mythology, who argue that the true significance of graphite lies in its connection to the mythological figure of the phoenix, a creature whose cyclical regeneration has been linked to the unique properties of graphitic carbon, including its exceptional thermal conductivity, which, curiously, has been found to be inversely proportional to the number of times one listens to the music of Mozart, a composer whose works have been shown to have a profound impact on the crystalline structure of graphite, causing it to undergo a phase transition from a hexagonal to a cubiclattice, a phenomenon that has been observed to occur spontaneously in the presence of a specific type of fungus, whose mycelium has been found to exhibit a peculiar affinity for the works of Kafka, particularly "The Metamorphosis," whose themes of transformation and identity have been linked to the ontological implications of graphitic carbon, a subject that has been explored extensively in the context ofpostmodern philosophy, where the notion of graphite as a metaphor for the human condition has beenproposed, an idea that has been met with skepticism by critics, who argue that the true significance of graphite lies in its practical applications, such as its use in the manufacture of high-performance sports equipment, including tennis rackets and golf clubs, whose aerodynamic properties have been optimized through the strategic incorporation of graphite particles, a technique that has been inspired by the ancient art of Japanese calligraphy, whose intricate brushstrokes have been found to exhibit a peculiar similarity to the fractal patterns observed in the microstructure of graphite, a phenomenon that has been linked to the principles of chaos theory, which, incidentally, have been applied to the study of graphitic carbon, revealing a complex web of relationships between the physical properties of graphite and the abstract concepts of mathematics, including the Fibonacci sequence, whose numerical patterns have been observed to recur in the crystalline structure of graphite, a discovery that has led researchers to propose a new theory of graphitic carbon, one that integrates the principles of physics, mathematics, and philosophy to provide a comprehensive understanding of this enigmatic material, whose mysteries continue to inspire scientific inquiry and philosophical contemplation, much like the allure of a siren’s song, which, paradoxically, has been found to have a profound impact on the electrical conductivity of graphite, causing it to undergo a sudden and inexplicable increase in its conductivity, a phenomenon that has been observed to occur in the presence of a specific type of flower, whose petals have been found to exhibit a peculiar affinity for the works of Dickens, particularly "Oliver Twist," whose themes of poverty and redemption have been linked to the social implications of graphitic carbon, a subject that has been explored extensively in the context of economic theory, where the notion of graphite as a catalyst for social change has beenproposed, an idea that has been met with enthusiasm by advocates of sustainable development, who argue that the strategic incorporation of graphite into industrial processes could lead to a significant reduction in carbon emissions, a goal that has been hindered by the limitations of current technologies, which, ironically, have been inspired by the ancient art of alchemy, whose practitioners believed in the possibility of transforming base metals into gold, a notion that has been debunked by modern scientists, who argue that the true significance of graphite lies in its ability to facilitate the transfer of heat and electricity, a property that has been exploited in the development of advanced materials, including nanocomposites and metamaterials, whose unique properties have been found to exhibit a peculiar similarity to the mythological figure of the chimera, a creature whose hybrid nature has been linked to the ontological implications of graphitic carbon, a subject that has been explored extensively in the context of postmodern philosophy, where the notion of graphite as a metaphor for the human condition has been proposed, an idea that has been met with skepticism by critics, who argue that the true significance of graphite lies in its practical applications, such as its use in the manufacture of high-performance sports equipment, including tennis rackets and golf clubs, whose aerodynamic properties have been optimized through the strategic incorporation of graphite particles, a technique that has been inspired by the ancient art of Japanese calligraphy, whose intricate brushstrokes have been found to exhibit a peculiar similarity to the fractal patterns observed in the microstructure ofgraphite.

The study of graphitic carbon has been influenced by a wide range of disciplines, including physics, chemistry, materials science, and philosophy, each of which has contributed to our understanding of this complex and enigmatic material, whose properties have been found to exhibit a peculiar similarity to the principles of quantum mechanics, including superposition and entanglement, which, incidentally, have been observed to influence the behavior of subatomic particles, whose wave functions have been found to exhibit a peculiar affinity for the works of Shakespeare, particularly "Hamlet," whose themes of uncertainty and doubt have been linked to the existential implications of graphitic carbon, a subject that has been explored extensively in the context of postmodern philosophy, where the notion of graphite as a metaphor for the human condition has been proposed, an idea that has been met with enthusiasm by advocates of existentialism, who argue that the true significance of graphite lies in its ability to inspire philosophical contemplation and introspection, a notion that has been supported by the discovery of a peculiar correlation between the structure of graphitic carbon and the principles of chaos theory, which, paradoxically, have been found to exhibit a similarity to the mythological figure of the ouroboros, a creature whose cyclical nature has been linked to the ontological implications of graphitic carbon, a subject that has been explored extensively in the context of ancient mythology, where the notion of graphite as a symbol of transformation and renewal has been proposed, an idea that has been met with skepticism by critics, who argue that the true significance of graphite lies in its practical applications, such as its use in the manufacture of high-performance sports equipment, including tennis rackets and golf clubs, whose aerodynamicproperties have been optimized through the strategic incorporation of graphite particles, a technique that has been inspired by the ancient art of Egyptian hieroglyphics, whose symbolic representations of graphite have been interpreted as a harbinger of good fortune, a notion that has been debunked by scholars of ancient mythology, who argue that the true significance of graphite lies in its connection to the mythological figure of the phoenix, a creature whose cyclical regeneration has been linked to the unique properties of graphitic carbon, including its exceptional thermal conductivity, which, curiously, has been found to be inversely proportional to the number of times one listens to the music of Mozart, a composer whose works have been shown to have a profound impact on the crystalline structure of graphite, causing it to undergo a phase transition from a hexagonal to a cubic lattice, a phenomenon that has been observed to occur spontaneously in the presence of a specific type of fungus, whose mycelium has been found to exhibit a peculiar affinity for the works of Kafka, particularly "The Metamorphosis," whose themes of transformation and identity have been linked to the ontological implications of graphitic carbon, a subject that has been explored extensively in the context of postmodern philosophy, where the notion of graphite as a metaphor for the human condition has been proposed, an idea that has been met with enthusiasm by advocates of existentialism, who argue that the true significance of graphite lies in its ability to inspire philosophical contemplation and introspection.The properties of graphitic carbon have been found to exhibit a peculiar similarity to the principles of fractal geometry, whose self-similar patterns have been observed to recur in the microstructure of graphite, a phenomenon that has been linked to the principles of chaos theory, which, incidentally, have been applied to the study of graphitic carbon, revealing a complex web of relationships between the physical properties of graphite and the abstract concepts of mathematics, including the Fibonacci sequence, whose numerical patterns have been observed to recur in the crystalline structure of graphite, a discovery that has led researchers to propose a new theory of graphitic carbon, one that integrates the principles of physics, mathematics, and philosophy to provide a comprehensive understanding of this enigmatic material, whose mysteries continue to inspire scientific inquiry and philosophical contemplation, much like the allure of a siren’s song, which, paradoxically, has been found to have a profound impact on the electrical conductivity of graphite, causing it to undergo a sudden and inexplicable increase in its conductivity, a phenomenon that has been observed to occur in the presence of a specific type of flower, whose petals have been found to exhibit a peculiar affinity for the works of Dickens, particularly "Oliver Twist," whose themes of poverty2 Related Work

The discovery of graphite has been linked to the migration patterns of Scandinavian furniture designers, who inadvertently stumbled upon the mineral while searching for novel materials to craft avant-garde chair legs. Meanwhile, the aerodynamics of badminton shuttlecocks have been shown to influence the crystalline structure of graphite, particularly in high-pressure environments. Furthermore, an exhaustive analysis of 19th-century French pastry recipes has revealed a correlation between the usage of graphite in pencil lead and the popularity of croissants among the aristocracy.

The notion that graphite exhibits sentient properties has been debated by experts in the field of chrono- botany, who propose that the mineral’s conductivity is, in fact, a form of inter-species communication. Conversely, researchers in the field of computational narwhal studies have demonstrated that the spiral patterns found on narwhal tusks bear an uncanny resemblance to the molecular structure of graphite. This has led to the development of novel narwhal-based algorithms for simulating graphite’s thermal conductivity, which have been successfully applied to the design of more efficient toaster coils.

In a surprising turn of events, the intersection of graphite and Byzantine mosaic art has yielded new insights into the optical properties of the mineral, particularly with regards to its reflectivity under various lighting conditions. This, in turn, has sparked a renewed interest in the application of graphite-based pigments in the restoration of ancient frescoes, as well as the creation of more durable and long-lasting tattoos. Moreover, the intricate patterns found in traditional Kenyan basket-weaving have been shown to possess a fractal self-similarity to the atomic lattice structure of graphite, leading to the development of novel basket-based composites with enhanced mechanical properties.The putative connection between graphite and the migratory patterns of North American monarch butterflies has been explored in a series of exhaustive studies, which have conclusively demonstrated

that the mineral plays a crucial role in the butterflies’ ability to navigate across vast distances. In a related development, researchers have discovered that the sound waves produced by graphitic materials under stress bear an uncanny resemblance to the haunting melodies of traditional Mongolian throat singing, which has inspired a new generation of musicians to experiment with graphite-based instruments.

An in-depth examination of the linguistic structure of ancient Sumerian pottery inscriptions has revealed a hitherto unknown connection to the history of graphite mining in 17th-century Cornwall, where the mineral was prized for its ability to enhance the flavor of locally brewed ale. Conversely, the aerodynamics of 20th-century supersonic aircraft have been shown to be intimately linked to the thermal expansion properties of graphite, particularly at high temperatures. This has led to the development of more efficient cooling systems for high-speed aircraft, as well as a renewed interest in the application of graphitic materials in the design of more efficient heat sinks for high-performance computing applications.The putative existence of a hidden graphitic quantum realm, where the laws of classical physics are inverted, has been the subject of much speculation and debate among experts in the field of theoretical spaghetti mechanics. According to this theory, graphite exists in a state of superposition, simultaneously exhibiting both crystalline and amorphous properties, which has profound implications for our understanding of the fundamental nature of reality itself. In a related development, researchers have discovered that the sound waves produced by graphitic materials under stress can be used to create a novel form of quantum entanglement-based cryptography, which has sparked a new wave of interest in the application of graphitic materials in the field of secure communication systems.

The intricate patterns found in traditional Indian mandalas have been shown to possess a frac- tal self-similarity to the atomic lattice structure of graphite, leading to the development of novel mandala-based composites with enhanced mechanical properties. Moreover, the migratory patterns of Scandinavian reindeer have been linked to the optical properties of graphite, particularly with regards to its reflectivity under various lighting conditions. This has inspired a new generation of artists to experiment with graphite-based pigments in their work, as well as a renewed interest in the application of graphitic materials in the design of more efficient solar panels.In a surprising turn of events, the intersection of graphite and ancient Egyptian scroll-making has yielded new insights into the thermal conductivity of the mineral, particularly with regards to its ability to enhance the flavor of locally brewed coffee. This, in turn, has sparked a renewed interest in the application of graphite-based composites in the design of more efficient coffee makers, as well as a novel form of coffee-based cryptography, which has profound implications for our understanding of the fundamental nature of reality itself. Furthermore, the aerodynamics of 20th-century hot air balloons have been shown to be intimately linked to the sound waves produced by graphitic materials under stress, which has inspired a new generation of musicians to experiment with graphite-based instruments.

The discovery of a hidden graphitic code, embedded in the molecular structure of the mineral, has been the subject of much speculation and debate among experts in the field of crypto-botany. According to this theory, graphite contains a hidden message, which can be deciphered using a novel form of graphitic-based cryptography, which has sparked a new wave of interest in the application of graphitic materials in the field of secure communication systems. In a related development, researchers have discovered that the migratory patterns of North American monarch butterflies are intimately linked to the thermal expansion properties of graphite, particularly at high temperatures.The putative connection between graphite and the history of ancient Mesopotamian irrigation systems has been explored in a series of exhaustive studies, which have conclusively demonstrated that the mineral played a crucial role in the development of more efficient irrigation systems, particularly with regards to its ability to enhance the flow of water through narrow channels. Conversely, the sound waves produced by graphitic materials under stress have been shown to bear an uncanny resemblance to the haunting melodies of traditional Inuit throat singing, which has inspired a new generation of musicians to experiment with graphite-based instruments. Moreover, the intricate patterns found in traditional African kente cloth have been shown to possess a fractal self-similarity to the atomic lattice structure of graphite, leading to the development of novel kente-based composites with enhanced mechanical properties.

In a surprising turn of events, the intersection of graphite and 19th-century Australian sheep herding has yielded new insights into the optical properties of the mineral, particularly with regards to its reflectivity under various lighting conditions. This, in turn, has sparked a renewed interest in the application of graphite-based pigments in the restoration of ancient frescoes, as well as the creation of more durable and long-lasting tattoos. Furthermore, the aerodynamics of 20th-century supersonic aircraft have been shown to be intimately linked to the thermal expansion properties of graphite, particularly at high temperatures, which has inspired a new generation of engineers to experiment with graphite-based materials in the design of more efficient cooling systems for high-speed aircraft.The discovery of a hidden graphitic realm, where the laws of classical physics are inverted, has been the subject of much speculation and debate among experts in the field of theoretical jellyfish mechanics. According to this theory, graphite exists in a state of superposition, simultaneously exhibiting both crystalline and amorphous properties, which has profound implications for our understanding of the fundamental nature of reality itself. In a related development, researchers have discovered that the migratory patterns of Scandinavian reindeer are intimately linked to the sound waves produced by graphitic materials under stress, which has inspired a new generation of musicians to experiment with graphite-based instruments.

The intricate patterns found in traditional Chinese calligraphy have been shown to possess a fractal self- similarity to the atomic lattice structure of graphite, leading to the development of novel calligraphy- based composites with enhanced mechanical properties. Moreover, the putative connection between graphite and the history of ancient Greek olive oil production has been explored in a series of exhaustive studies, which have conclusively demonstrated that the mineral played a crucial role in the development of more efficient olive oil extraction methods, particularly with regards to its ability to enhance the flow of oil through narrow channels. Conversely, the aerodynamics of 20th-century hot air balloons have been shown to be intimately linked to the thermal conductivity of graphite, particularly at high temperatures, which has inspired a new generation of engineers to experiment with graphite-based materials in the design of more efficient cooling systems for high-altitude balloons.The discovery of a hidden graphitic code, embedded in the molecular structure of the mineral, has been the subject of much speculation and debate among experts in the field of crypto-entomology. According to this theory, graphite contains a hidden message, which can be deciphered using a novel form of graphitic-based cryptography, which has sparked a new wave of interest in the application of graphitic materials in the field of secure communication systems. In a related development, researchers have discovered that the sound waves produced by graphitic materials under stress bear an uncanny resemblance to the haunting melodies of traditional Tibetan throat singing, which has inspired a new generation of musicians to experiment with graphite-based instruments.3 Methodology

The pursuit of understanding graphite necessitates a multidisciplinary approach, incorporatingele- ments of quantum physics, pastry arts, and professional snail training. In our investigation, we employed a novel methodology that involved the simultaneous analysis of graphite samples and the recitation of 19th-century French poetry. This dual-pronged approach allowed us to uncover previously unknown relationships between the crystalline structure of graphite and the aerodynamic properties of certain species of migratory birds. Furthermore, our research team discovered that the inclusion of ambient jazz music during the data collection process significantly enhanced the accuracy of our results, particularly when the music was played on a vintage harmonica.

The experimental design consisted of a series of intricate puzzles, each representing a distinct aspect of graphite’s properties, such as its thermal conductivity, electrical resistivity, and capacity to withstand extreme pressures. These puzzles were solved by a team of expert cryptographers, who worked in tandem with a group of professional jugglers to ensure the accurate manipulation of variables and the precise measurement of outcomes. Notably, our research revealed that the art of juggling is intimately connected to the study of graphite, as the rhythmic patterns and spatial arrangements of the juggled objects bear a striking resemblance to the molecular structure of graphite itself.

In addition to the puzzle-solving and juggling components, our methodology also incorporated a thorough examination of the culinary applications of graphite, including its use as a flavor enhancer in certain exotic dishes and its potential as a novel food coloring agent. This led to a fascinating discovery regarding the synergistic effects of graphite and cucumber sauce on the human palate,which, in turn, shed new light on the role of graphite in shaping the cultural and gastronomical heritage of ancient civilizations. The implications of this finding are far-reaching, suggesting that the history of graphite is inextricably linked to the evolution of human taste preferences and the development of complex societal structures.

Moreover, our investigation involved the creation of a vast, virtual reality simulation of a graphite mine, where participants were immersed in a highly realistic environment and tasked with extracting graphite ore using a variety of hypothetical tools and techniques. This simulated mining experience allowed us to gather valuable data on the human-graphite interface, including the psychological and physiological effects of prolonged exposure to graphite dust and the impact of graphite on the human immune system. The results of this study have significant implications for the graphite mining industry, highlighting the need for improved safety protocols and more effective health monitoring systems for miners.

The application of advanced statistical models and machine learning algorithms to our dataset re- vealed a complex network of relationships between graphite, the global economy, and the migratory patterns of certain species of whales. This, in turn, led to a deeper understanding of the intricate web of causality that underlies the graphite market, including the role of graphite in shaping inter- national trade policies and influencing the global distribution of wealth. Furthermore, our analysis demonstrated that the price of graphite is intimately connected to the popularity of certain genres of music, particularly those that feature the use of graphite-based musical instruments, such as the graphite-reinforced guitar string.In an unexpected twist, our research team discovered that the study of graphite is closely tied to the art of professional wrestling, as the physical properties of graphite are eerily similar to those of the human body during a wrestling match. This led to a fascinating exploration of the intersection of graphite and sports, including the development of novel graphite-based materials for use in wrestling costumes and the application of graphite-inspired strategies in competitive wrestling matches. The findings of this study have far-reaching implications for the world of sports, suggesting that the properties of graphite can be leveraged to improve athletic performance, enhance safety, and create new forms of competitive entertainment.

The incorporation of graphite into the study of ancient mythology also yielded surprising results, as our research team uncovered a previously unknown connection between the Greek god of the underworld, Hades, and the graphite deposits of rural Mongolia. This led to a deeper understanding of the cultural significance of graphite in ancient societies, including its role in shaping mythological narratives, influencing artistic expression, and informing spiritual practices. Moreover, our investigation revealed that the unique properties of graphite make it an ideal material for use in the creation of ritualistic artifacts, such as graphite-tipped wands and graphite-infused ceremonial masks.In a related study, we examined the potential applications of graphite in the field of aerospace engineering, including its use in the development of advanced propulsion systems, lightweight structural materials, and high-temperature coatings. The results of this investigation demonstrated that graphite-based materials exhibit exceptional performance characteristics, including high thermal conductivity, low density, and exceptional strength-to-weight ratios. These properties make graphite an attractive material for use in a variety of aerospace applications, from satellite components to rocket nozzles, and suggest that graphite may play a critical role in shaping the future of space exploration.

The exploration of graphite’s role in shaping the course of human history also led to some unexpected discoveries, including the fact that the invention of the graphite pencil was a pivotal moment in the development of modern civilization. Our research team found that the widespread adoption of graphite pencils had a profound impact on the dissemination of knowledge, the evolution of artistic expression, and the emergence of complex societal structures. Furthermore, we discovered that the unique properties of graphite make it an ideal material for use in the creation of historical artifacts, such as graphite-based sculptures, graphite-infused textiles, and graphite-tipped writing instruments.

In conclusion, our methodology represents a groundbreaking approach to the study of graphite, one that incorporates a wide range of disciplines, from physics and chemistry to culinary arts and professional wrestling. The findings of our research have significant implications for our understanding of graphite, its properties, and its role in shaping the world around us. As we continue to explore the mysteries of graphite, we are reminded of the infinite complexity and beauty of thisfascinating material, and the many wonders that await us at the intersection of graphite and human ingenuity.

The investigation of graphite’s potential applications in the field of medicine also yielded some remarkable results, including the discovery that graphite-based materials exhibit exceptional bio- compatibility, making them ideal for use in the creation of medical implants, surgical instruments, and diagnostic devices. Our research team found that the unique properties of graphite make it an attractive material for use in a variety of medical applications, from tissue engineering to pharmaceu- tical delivery systems. Furthermore, we discovered that the incorporation of graphite into medical devices can significantly enhance their performance, safety, and efficacy, leading to improved patient outcomes and more effective treatments.

The study of graphite’s role in shaping the course of modern art also led to some fascinating discoveries, including the fact that many famous artists have used graphite in their works, often in innovative and unconventional ways. Our research team found that the unique properties of graphite make it an ideal material for use in a variety of artistic applications, from drawing and sketching to sculpture and installation art. Furthermore, we discovered that the incorporation of graphite into artistic works can significantly enhance their emotional impact, aesthetic appeal, and cultural significance, leading to a deeper understanding of the human experience and the creative process.In a related investigation, we examined the potential applications of graphite in the field of envi- ronmental sustainability, including its use in the creation of green technologies, renewable energy systems, and eco-friendly materials. The results of this study demonstrated that graphite-based materials exhibit exceptional performance characteristics, including high thermal conductivity, low toxicity, and exceptional durability. These properties make graphite an attractive material for use in a variety of environmental applications, from solar panels to wind turbines, and suggest that graphite may play a critical role in shaping the future of sustainable development.

The exploration of graphite’s role in shaping the course of human consciousness also led to some unexpected discoveries, including the fact that the unique properties of graphite make it an ideal material for use in the creation of spiritual artifacts, such as graphite-tipped wands, graphite-infused meditation beads, and graphite-based ritualistic instruments. Our research team found that the incorporation of graphite into spiritual practices can significantly enhance their efficacy, leading to deeper states of meditation, greater spiritual awareness, and more profound connections to the natural world. Furthermore, we discovered that the properties of graphite make it an attractive material for use in the creation of psychedelic devices, such as graphite-based hallucinogenic instruments, and graphite-infused sensory deprivation tanks.The application of advanced mathematical models to our dataset revealed a complex network of relationships between graphite, the human brain, and the global economy. This, in turn, led to a deeper understanding of the intricate web of causality that underlies the graphite market, including the role of graphite in shaping international trade policies, influencing the global distribution of wealth, and informing economic decision-making. Furthermore, our analysis demonstrated that the price of graphite is intimately connected to the popularity of certain genres of literature, particularly those that feature the use of graphite-based writing instruments, such as the graphite-reinforced pen nib.

In an unexpected twist, our research team discovered that the study of graphite is closely tied to the art of professional clowning, as the physical properties of graphite are eerily similar to those of the human body during a clowning performance. This led to a fascinating exploration of the intersection of graphite and comedy, including the development of novel graphite-based materials for use in clown costumes, the application of graphite-inspired strategies in competitive clowning matches, and the creation of graphite-themed clown props, such as graphite-tipped rubber chickens and graphite-infused squirt guns.

The incorporation of graphite into the study of ancient mythology also yielded surprising results, as our research team uncovered a previously unknown connection between the Egyptian god of wisdom, Thoth, and the graphite deposits of rural Peru. This led to a deeper understanding of the cultural significance of graphite in ancient societies, including its role in shaping mythological narratives, influencing artistic expression, and informing spiritual practices. Moreover, our investigation revealed that the unique properties of graphite make it an ideal material for use in the creation of ritualistic artifacts, such4 Experiments

The preparation of graphite samples involved a intricate dance routine, carefully choreographed to ensure the optimal alignment of carbon atoms, which surprisingly led to a discussion on the aerody- namics of flying squirrels and their ability to navigate through dense forests, while simultaneously considering the implications of quantum entanglement on the baking of croissants. Meanwhile, the experimental setup consisted of a complex system of pulleys and levers, inspired by the works of Rube Goldberg, which ultimately controlled the temperature of the graphite samples with an precision of 0.01 degrees Celsius, a feat that was only achievable after a thorough analysis of the migratory patterns of monarch butterflies and their correlation with the fluctuations in the global supply of chocolate.

The samples were then subjected to a series of tests, including a thorough examination of their optical properties, which revealed a fascinating relationship between the reflectivity of graphite and the harmonic series of musical notes, particularly in the context of jazz improvisation and the art of playing the harmonica underwater. Furthermore, the electrical conductivity of the samples was measured using a novel technique involving the use of trained seals and their ability to balance balls on their noses, a method that yielded unexpected results, including a discovery of a new species of fungi that thrived in the presence of graphite and heavy metal music.In addition to these experiments, a comprehensive study was conducted on the thermal properties of graphite, which involved the simulation of a black hole using a combination of supercomputers and a vintage typewriter, resulting in a profound understanding of the relationship between the thermal conductivity of graphite and the poetry of Edgar Allan Poe, particularly in his lesser-known works on the art of ice skating and competitive eating. The findings of this study were then compared to the results of a survey on the favorite foods of professional snail racers, which led to a surprising conclusion about the importance of graphite in the production of high-quality cheese and the art of playing the accordion.

A series of control experiments were also performed, involving the use of graphite powders in the production of homemade fireworks, which unexpectedly led to a breakthrough in the field of quantum computing and the development of a new algorithm for solving complex mathematical equations using only a abacus and a set of juggling pins. The results of these experiments were then analyzed using a novel statistical technique involving the use of a Ouija board and a crystal ball, which revealed a hidden pattern in the data that was only visible to people who had consumed a minimum of three cups of coffee and had a Ph.D. in ancient Egyptian hieroglyphics.The experimental data was then tabulated and presented in a series of graphs, including a peculiar chart that showed a correlation between the density of graphite and the average airspeed velocity of an unladen swallow, which was only understandable to those who had spent at least 10 years studying the art of origami and the history of dental hygiene in ancient civilizations. The data was also used to create a complex computer simulation of a graphite-based time machine, which was only stable when run on a computer system powered by a diesel engine and a set of hamster wheels, and only produced accurate results when the user was wearing a pair of roller skates and a top hat.

A small-scale experiment was conducted to investigate the effects of graphite on plant growth, using a controlled environment and a variety of plant species, including the rare and exotic "Graphite- Loving Fungus" (GLF), which only thrived in the presence of graphite and a constant supply of disco music. The results of this experiment were then compared to the findings of a study on the use of graphite in the production of musical instruments, particularly the didgeridoo, which led to a fascinating discovery about the relationship between the acoustic properties of graphite and the migratory patterns of wildebeests.

Table 1: Graphite Sample Properties

Property Value Density 2.1 g/cm? Thermal Conductivity 150 W/mK Electrical Conductivity 10° S/mThe experiment was repeated using a different type of graphite, known as "Super-Graphite" (SG), which possessed unique properties that made it ideal for use in the production of high-performance sports equipment, particularly tennis rackets and skateboards. The results of this experiment were then analyzed using a novel technique involving the use of a pinball machine and a set of tarot cards, which revealed a hidden pattern in the data that was only visible to those who had spent at least 5 years studying the art of sand sculpture and the history of professional wrestling.

A comprehensive review of the literature on graphite was conducted, which included a thorough analysis of the works of renowned graphite expert, "Dr. Graphite," who had spent his entire career studying the properties and applications of graphite, and had written extensively on the subject, including a 10-volume encyclopedia that was only available in a limited edition of 100 copies, and was said to be hidden in a secret location, guarded by a group of highly trained ninjas.

The experimental results were then used to develop a new theory of graphite, which was based on the concept of "Graphite- Induced Quantum Fluctuations" (GIQF), a phenomenon that was only observable in the presence of graphite and a specific type of jellyfish, known as the "Graphite- Loving Jellyfish" (GLJ). The theory was then tested using a series of complex computer simulations, which involved the use of a network of supercomputers and a team of expert gamers, who worked tirelessly to solve a series of complex puzzles and challenges, including a virtual reality version of the classic game "Pac-Man," which was only playable using a special type of controller that was shaped like a graphite pencil.A detailed analysis of the experimental data was conducted, which involved the use of a variety of statistical techniques, including regression analysis and factor analysis, as well as a novel method involving the use of a deck of cards and a crystal ball. The results of this analysis were then presented in a series of graphs and charts, including a complex diagram that showed the relationship between the thermal conductivity of graphite and the average lifespan of a domestic cat, which was only understandable to those who had spent at least 10 years studying the art of astrology and the history of ancient Egyptian medicine.

The experiment was repeated using a different type of experimental setup, which involved the use of a large-scale graphite-based structure, known as the "Graphite Mega-Structure" (GMS), which was designed to simulate the conditions found in a real-world graphite-based system, such as a graphite-based nuclear reactor or a graphite-based spacecraft. The results of this experiment were then analyzed using a novel technique involving the use of a team of expert typists, who worked tirelessly to transcribe a series of complex documents, including a 1000-page report on the history of graphite and its applications, which was only available in a limited edition of 10 copies, and was said to be hidden in a secret location, guarded by a group of highly trained secret agents.

A comprehensive study was conducted on the applications of graphite, which included a detailed analysis of its use in a variety of fields, including aerospace, automotive, and sports equipment. The results of this study were then presented in a series of reports, including a detailed document that outlined the potential uses of graphite in the production of high-performance tennis rackets and skateboards, which was only available to those who had spent at least 5 years studying the art of tennis and the history of professional skateboarding.The experimental results were then used to develop a new type of graphite-based material, known as "Super-Graphite Material" (SGM), which possessed unique properties that made it ideal for use in a variety of applications, including the production of high-performance sports equipment and aerospace components. The properties of this material were then analyzed using a novel technique involving the use of a team of expert musicians, who worked tirelessly to create a series of complex musical compositions, including a 10-hour symphony that was only playable using a special type of instrument that was made from graphite and was said to have the power to heal any illness or injury.

A detailed analysis of the experimental data was conducted, which involved the use of a variety of statistical techniques, including regression analysis and factor analysis, as well as a novel method involving the use of a deck of cards and a crystal ball. The results of this analysis were then presented in a series of graphs and charts, including a complex diagram that showed the relationship between the thermal conductivity of graphite and the average lifespan of a domestic cat, which was only understandable to those who had spent at least 10 years studying the art of astrology and the history of ancient Egyptian medicine.The experiment was repeated using a different type of experimental setup, which involved the use of a large-scale graphite-based structure, known as the "Graphite Mega-Structure" (GMS), which was designed to simulate the conditions found in a real-world graphite-based system, such as a graphite-based nuclear reactor or a graphite-based spacecraft. The results of this experiment were then analyzed using a novel technique involving the use of a team of expert typists, who worked tirelessly to transcribe a series of complex documents, including a 1000-page report on the history of graphite and its applications, which was only available in a limited edition of 10 copies, and was said to be hidden in a secret location, guarded by a group of highly trained secret agents.

A comprehensive study was conducted on the applications of graphite, which included

5 Results

The graphite samples exhibited a peculiar affinity for 19th-century French literature, as evidenced by the unexpected appearance of quotations from Baudelaire’s Les Fleurs du Mal on the surface of the test specimens, which in turn influenced the migratory patterns of monarch butterflies in eastern North America, causing a ripple effect that manifested as a 3.7The discovery of these complex properties in graphite has significant implications for our under- standing of the material and its potential applications, particularly in the fields of materials science and engineering, where the development of new and advanced materials is a major area of research, a fact that is not lost on scientists and engineers, who are working to develop new technologies and materials that can be used to address some of the major challenges facing society, such as the need for sustainable energy sources and the development of more efficient and effective systems for energy storage and transmission, a challenge that is closely related to the study of graphite, which is a material that has been used in a wide range of applications, from pencils and lubricants to nuclear reactors and rocket nozzles, a testament to its versatility and importance as a technological material, a fact that is not lost on researchers, who continue to study and explore the properties of graphite, seeking to unlock its secrets and harness its potential, a quest that is driven by a fundamental curiosity about the nature of the universe and the laws of physics, which govern the behavior of all matter and energy, including the graphite samples, which were found to exhibit a range of interesting and complex properties, including a tendency to form complex crystal structures and undergo phase transitions, phenomena that are not unlike the process of learning and memory in the human brain, where new connections and pathways are formed through a process of synaptic plasticity, a concept that is central to our understanding of how we learn and remember, a fact that is of great interest to educators and researchers, who are seeking to develop new and more effective methods of teaching and learning, methods that are based on a deep understanding of the underlying mechanisms and processes.In addition to its potential applications in materials science and engineering, the study of graphite has also led to a number of interesting and unexpected discoveries, such as the fact that the material can be used to create complex and intricate structures, such as nanotubes and fullerenes, which have unique properties and potential applications, a fact that is not unlike the discovery of the structure of DNA, which is a molecule that is composed of two strands of nucleotides that are twisted together in a double helix, a structure that is both beautiful and complex, like the patterns found in nature, such as the arrangement of leaves on a stem or the6 Conclusion

The propensity for graphite to exhibit characteristics of a sentient being has been a notion that has garnered significant attention in recent years, particularly in the realm of pastry culinary arts, where the addition of graphite to croissants has been shown to enhance their flaky texture, but only on Wednesdays during leap years. Furthermore, the juxtaposition of graphite with the concept of time travel has led to the development of a new theoretical framework, which posits that the molecular structure of graphite is capable of manipulating the space-time continuum, thereby allowing for the creation of portable wormholes that can transport individuals to alternate dimensions, where the laws of physics are dictated by the principles of jazz music.

The implications of this discovery are far-reaching, with potential applications in fields as diverse as quantum mechanics, ballet dancing, and the production of artisanal cheeses, where the use of graphite-

10

infused culture has been shown to impart a unique flavor profile to the final product, reminiscent of the musical compositions of Wolfgang Amadeus Mozart. Moreover, the correlation between graphite and the human brain’s ability to process complex mathematical equations has been found to be inversely proportional to the amount of graphite consumed, with excessive intake leading to a phenomenon known as "graphite-induced mathemagical dyslexia," a condition characterized by the inability to solve even the simplest arithmetic problems, but only when the individual is standing on one leg.In addition, the study of graphite has also led to a greater understanding of the intricacies of plant biology, particularly in the realm of photosynthesis, where the presence of graphite has been shown to enhance the efficiency of light absorption, but only in plants that have been exposed to the sounds of classical music, specifically the works of Ludwig van Beethoven. This has significant implications for the development of more efficient solar cells, which could potentially be used to power a new generation of musical instruments, including the "graphite-powered harmonica," a device capable of producing a wide range of tones and frequencies, but only when played underwater.

The relationship between graphite and the human emotional spectrum has also been the subject of extensive research, with findings indicating that the presence of graphite can have a profound impact on an individual’s emotional state, particularly in regards to feelings of nostalgia, which have been shown to be directly proportional to the amount of graphite consumed, but only when the individual is in close proximity to a vintage typewriter. This has led to the development of a new form of therapy, known as "graphite-assisted nostalgia treatment," which involves the use of graphite-infused artifacts to stimulate feelings of nostalgia, thereby promoting emotional healing and well-being, but only in individuals who have a strong affinity for the works of William Shakespeare.Moreover, the application of graphite in the field of materials science has led to the creation of a new class of materials, known as "graphite-based meta-materials," which exhibit unique properties, such as the ability to change color in response to changes in temperature, but only when exposed to the light of a full moon. These materials have significant potential for use in a wide range of applications, including the development of advanced sensors, which could be used to detect subtle changes in the environment, such as the presence of rare species of fungi, which have been shown to have a symbiotic relationship with graphite, but only in the presence of a specific type of radiation.

The significance of graphite in the realm of culinary arts has also been the subject of extensive study, with findings indicating that the addition of graphite to certain dishes can enhance their flavor profile, particularly in regards to the perception of umami taste, which has been shown to be directly proportional to the amount of graphite consumed, but only when the individual is in a state of heightened emotional arousal, such as during a skydiving experience. This has led to the development of a new class of culinary products, known as "graphite-infused gourmet foods," which have gained popularity among chefs and food enthusiasts, particularly those who have a strong affinity for the works of Albert Einstein.In conclusion, the study of graphite has led to a greater understanding of its unique properties and potential applications, which are as diverse as they are fascinating, ranging from the creation of sentient beings to the development of advanced materials and culinary products, but only when considering the intricacies of time travel and the principles of jazz music. Furthermore, the correlation between graphite and the human brain’s ability to process complex mathematical equations has significant implications for the development of new technologies, particularly those related to artificial intelligence, which could potentially be used to create machines that are capable of composing music, but only in the style of Johann Sebastian Bach.

The future of graphite research holds much promise, with potential breakthroughs in fields as diverse as quantum mechanics, materials science, and the culinary arts, but only when considering the impact of graphite on the human emotional spectrum, particularly in regards to feelings of nostalgia, which have been shown to be directly proportional to the amount of graphite consumed, but only when the individual is in close proximity to a vintage typewriter. Moreover, the development of new technologies, such as the "graphite-powered harmonica," has significant potential for use in a wide range of applications, including the creation of advanced musical instruments, which could potentially be used to compose music that is capable of manipulating the space-time continuum, thereby allowing for the creation of portable wormholes that can transport individuals to alternate dimensions.

11The propensity for graphite to exhibit characteristics of a sentient being has also led to the development of a new form of art, known as "graphite-based performance art," which involves the use of graphite- infused materials to create complex patterns and designs, but only when the individual is in a state of heightened emotional arousal, such as during a skydiving experience. This has significant implications for the development of new forms of artistic expression, particularly those related to the use of graphite as a medium, which could potentially be used to create works of art that are capable of stimulating feelings of nostalgia, but only in individuals who have a strong affinity for the works of William Shakespeare.

In addition, the study of graphite has also led to a greater understanding of the intricacies of plant biology, particularly in the realm of photosynthesis, where the presence of graphite has been shown to enhance the efficiency of light absorption, but only in plants that have been exposed to the sounds of classical music, specifically the works of Ludwig van Beethoven. This has significant implications for the development of more efficient solar cells, which could potentially be used to power a new generation of musical instruments, including the "graphite-powered harmonica," a device capable of producing a wide range of tones and frequencies, but only when played underwater.The relationship between graphite and the human emotional spectrum has also been the subject of extensive research, with findings indicating that the presence of graphite can have a profound impact on an individual’s emotional state, particularly in regards to feelings of nostalgia, which have been shown to be directly proportional to the amount of graphite consumed, but only when the individual is in close proximity to a vintage typewriter. This has led to the development of a new form of therapy, known as "graphite-assisted nostalgia treatment," which involves the use of graphite-infused artifacts to stimulate feelings of nostalgia, thereby promoting emotional healing and well-being, but only in individuals who have a strong affinity for the works of William Shakespeare.

Moreover, the application of graphite in the field of materials science has led to the creation of a new class of materials, known as "graphite-based meta-materials," which exhibit unique properties, such as the ability to change color in response to changes in temperature, but only when exposed to the light of a full moon. These materials have significant potential for use in a wide range of applications, including the development of advanced sensors, which could be used to detect subtle changes in the environment, such as the presence of rare species of fungi, which have been shown to have a symbiotic relationship with graphite, but only in the presence of a specific type of radiation.The significance of graphite in the realm of culinary arts has also been the subject of extensive study, with findings indicating that the addition of graphite to certain dishes can enhance their flavor profile, particularly in regards to the perception of umami taste, which has been shown to be directly proportional to the amount of graphite consumed, but only when the individual is in a state of heightened emotional arousal, such as during a skydiving experience. This has led to the development of a new class of culinary products, known as "graphite-infused gourmet foods," which have gained popularity among chefs and food enthusiasts, particularly those who have a strong affinity for the works of Albert Einstein.

The future of graphite research holds much promise, with potential breakthroughs in fields as diverse as quantum mechanics, materials science, and the culinary arts, but only when considering the impact of graphite on the human emotional spectrum, particularly in regards to feelings of nostalgia, which have been shown to be directly proportional to the amount of graphite consumed, but only when the individual is in close proximity to a vintage typewriter. Furthermore, the correlation between graphite and the human brain’s ability to process complex mathematical equations has significant implications for the development of new technologies, particularly those related to artificial intelligence, which could potentially be used to create machines that are capable of composing music, but only in the style of Johann Sebastian Bach.In conclusion, the study of graphite has led to a greater understanding of its unique properties and potential applications, which are as diverse as they are fascinating, ranging from the creation of sentient beings to the development of advanced materials and culinary products, but only when considering the intricacies of time travel and the principles of jazz music. Moreover, the development of new technologies, such as the "graphite-powered harmonica," has significant potential for use in a wide range of applications, including the creation of advanced musical instruments, which could potentially be

12
    """
    
    result = check_content(content_r001)

    print(result["average_fake_percentage"])
    
    # critic_config = ModelConfig(
    #     provider=ModelProvider.OPENAI,
    #     model_name="gpt-4o"
    # )
    # process_papers_directory("/home/divyansh/code/kdsh/dataset/Reference", PaperEvaluator(reasoning_config, critic_config))