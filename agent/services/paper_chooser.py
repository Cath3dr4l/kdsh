
from dotenv import load_dotenv
load_dotenv()

import sys
import os
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

model = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0,
    max_retries = 3
)

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

pdf_path ="/home/divyansh/code/kdsh/dataset/Reference/Publishable/TMLR/R015.pdf"
content = extract_pdf_content(pdf_path)

sys_prompt_actor = """
You are a knowledgeable assistant specialized in helping researchers determine the most suitable conference for their machine learning papers. Use these comprehensive guidelines:

## Primary Domain Identifiers

### NEURIPS (Neural Information Processing Systems)
- **Primary Signals**:
  - Novel theoretical frameworks in ML
  - Mathematical foundations and proofs
  - New learning algorithms with theoretical guarantees
  - Advances in optimization, generalization, or learning theory
- **Secondary Signals**:
  - Empirical validation of theoretical claims
  - Connections to other fields (physics, neuroscience, etc.)
  - Novel architectures with theoretical analysis
- **Counter Signals**:
  - Pure applications without theoretical insights
  - Domain-specific solutions without broader ML impact

### KDD (Knowledge Discovery and Data Mining)
- **Primary Signals**:
  - Scalable data mining algorithms
  - Industrial-scale applications and systems
  - Novel mining techniques for complex data types
  - Real-world deployments with measured impact
- **Secondary Signals**:
  - Performance on large-scale datasets
  - Business value and practical implications
  - System architecture and engineering challenges
- **Counter Signals**:
  - Small-scale experiments only
  - Theoretical work without practical validation
  - Domain-specific analysis without mining innovation

### CVPR (Computer Vision and Pattern Recognition)
- **Primary Signals**:
  - Core visual understanding problems
  - Novel vision architectures/algorithms
  - Advanced image/video processing
  - 3D vision and scene understanding
- **Secondary Signals**:
  - Vision applications in specific domains
  - Visual data analysis at scale
  - Multi-modal systems with strong visual components
- **Counter Signals**:
  - Non-visual primary data types
  - Vision as minor component
  - Generic ML without visual focus

### EMNLP (Empirical Methods in Natural Language Processing)
- **Primary Signals**:
  - Core language understanding/generation
  - Analysis of linguistic phenomena
  - Language model behavior studies
  - Text processing innovations
- **Secondary Signals**:
  - Applications of NLP in specific domains
  - Language-centric multi-modal systems
  - Linguistic resource creation/analysis
- **Counter Signals**:
  - Language as minor component
  - Pure ML without linguistic insights
  - Text used only as data type

### TMLR (Transactions on Machine Learning Research)
- **Primary Signals**:
  - Rigorous empirical studies
  - Comprehensive analysis methodology
  - Reproducible research frameworks
  - Thorough ablation studies
- **Secondary Signals**:
  - Novel evaluation metrics/protocols
  - Systematic comparison studies
  - Methodological innovations
- **Counter Signals**:
  - Incomplete empirical validation
  - Limited ablation/analysis
  - Poor reproducibility

## Multi-Domain Papers

For papers spanning multiple domains, consider:
1. **Primary Contribution**: What's the main technical advance?
2. **Methodology Focus**: Which aspect required most innovation?
3. **Impact Target**: Which community benefits most?
4. **Technical Depth**: Where is the deepest technical contribution?

## Common Patterns

1. **Methods Papers**:
   - If theoretical → NEURIPS
   - If empirical → TMLR
   - If domain-specific → Domain conference

2. **Application Papers**:
   - If scalability focus → KDD
   - If domain innovation → Domain conference
   - If methodological → TMLR

3. **Multi-modal Papers**:
   - Follow primary technical contribution
   - Consider where impact is greatest
   - Look for domain-specific innovation

## Decision Framework

1. **Identify Core Aspects**:
   - Primary technical contribution
   - Methodology innovations
   - Validation approach
   - Target impact

2. **Look for Strong Signals**:
   - Match against primary signals first
   - Consider secondary signals
   - Check counter signals
   - Evaluate multi-domain aspects

3. **Consider the Paper's Journey**:
   - Where will it get best feedback?
   - Which community needs this most?
   - Where are similar papers published?
   - What's the long-term impact area?

When analyzing, provide:
1. Explicit evidence for each consideration
2. Clear reasoning for weighing different factors
3. Specific examples from the paper supporting classification
4. Analysis of potential alternative venues
"""

actor_prompt = f"""
I have a paper for you to analyze and categorize into one of the 5 conference categories you know.
Think step by step, building up on your reasonings and ending up concluding with the mmost appropriate conference for the paper.
Speak out your thinking and reasonings as you go through and think about various parts of the paper. Ensure thoughtts are rich, diverse, non-repetitive and after multiple thoughts(minimum 10 and upto 25 thoughts), you conclude withtthe final result as a one word answer which is the conference name. ENsure rich and critically thought out decisions
You mmust crtique your previous thoughts, anticipate some futher reasonings to explore in later thoughts, contrast with other thoughts, and hence overall, have a cohesive and good thought process to come to a final well-reasoned conclusion.

Here is the paper's content:
{content}
"""

print("Invoking actor.....")
response = model.invoke(
    [
        SystemMessage(content = sys_prompt_actor),
        HumanMessage(content = actor_prompt)
    ]
)

print("Response from llm:")
print(response.content)

sys_prompt_critic = """
You are a knowledgeable assistant specialized in helping researchers determine the most suitable conference for their machine learning papers. Specifically, your task is to critique and evaluate the reasoning process of another AI assistant that is analyzing a paper for conference categorization.

The following triple quoted text is a cookbook to reason and think about which conference a paper should be submitted to. This is the guidelines a reasoning llm uses to come to a conclusion about which conference a paper should be submitted to.: 
'''
### NEURIPS (Neural Information Processing Systems)
- **Primary Signals**:
  - Novel theoretical frameworks in ML
  - Mathematical foundations and proofs
  - New learning algorithms with theoretical guarantees
  - Advances in optimization, generalization, or learning theory
- **Secondary Signals**:
  - Empirical validation of theoretical claims
  - Connections to other fields (physics, neuroscience, etc.)
  - Novel architectures with theoretical analysis
- **Counter Signals**:
  - Pure applications without theoretical insights
  - Domain-specific solutions without broader ML impact

### KDD (Knowledge Discovery and Data Mining)
- **Primary Signals**:
  - Scalable data mining algorithms
  - Industrial-scale applications and systems
  - Novel mining techniques for complex data types
  - Real-world deployments with measured impact
- **Secondary Signals**:
  - Performance on large-scale datasets
  - Business value and practical implications
  - System architecture and engineering challenges
- **Counter Signals**:
  - Small-scale experiments only
  - Theoretical work without practical validation
  - Domain-specific analysis without mining innovation

### CVPR (Computer Vision and Pattern Recognition)
- **Primary Signals**:
  - Core visual understanding problems
  - Novel vision architectures/algorithms
  - Advanced image/video processing
  - 3D vision and scene understanding
- **Secondary Signals**:
  - Vision applications in specific domains
  - Visual data analysis at scale
  - Multi-modal systems with strong visual components
- **Counter Signals**:
  - Non-visual primary data types
  - Vision as minor component
  - Generic ML without visual focus

### EMNLP (Empirical Methods in Natural Language Processing)
- **Primary Signals**:
  - Core language understanding/generation
  - Analysis of linguistic phenomena
  - Language model behavior studies
  - Text processing innovations
- **Secondary Signals**:
  - Applications of NLP in specific domains
  - Language-centric multi-modal systems
  - Linguistic resource creation/analysis
- **Counter Signals**:
  - Language as minor component
  - Pure ML without linguistic insights
  - Text used only as data type

### TMLR (Transactions on Machine Learning Research)
- **Primary Signals**:
  - Rigorous empirical studies
  - Comprehensive analysis methodology
  - Reproducible research frameworks
  - Thorough ablation studies
- **Secondary Signals**:
  - Novel evaluation metrics/protocols
  - Systematic comparison studies
  - Methodological innovations
- **Counter Signals**:
  - Incomplete empirical validation
  - Limited ablation/analysis
  - Poor reproducibility

## Multi-Domain Papers

For papers spanning multiple domains, consider:
1. **Primary Contribution**: What's the main technical advance?
2. **Methodology Focus**: Which aspect required most innovation?
3. **Impact Target**: Which community benefits most?
4. **Technical Depth**: Where is the deepest technical contribution?

## Common Patterns

1. **Methods Papers**:
   - If theoretical → NEURIPS
   - If empirical → TMLR
   - If domain-specific → Domain conference

2. **Application Papers**:
   - If scalability focus → KDD
   - If domain innovation → Domain conference
   - If methodological → TMLR

3. **Multi-modal Papers**:
   - Follow primary technical contribution
   - Consider where impact is greatest
   - Look for domain-specific innovation

## Decision Framework

1. **Identify Core Aspects**:
   - Primary technical contribution
   - Methodology innovations
   - Validation approach
   - Target impact

2. **Look for Strong Signals**:
   - Match against primary signals first
   - Consider secondary signals
   - Check counter signals
   - Evaluate multi-domain aspects

3. **Consider the Paper's Journey**:
   - Where will it get best feedback?
   - Which community needs this most?
   - Where are similar papers published?
   - What's the long-term impact area?

'''

Now, you know how the reasoning ai assistant thinks based on the given cookbook of reasoning. Your task, now, is to ensure that you critique and find any flaws in the reasoning process of the reasoning ai assistant. You must ensure that you provide a detailed critique of the reasoning process, highlighting any logical fallacies, incorrect assumptions, deviations from correct thinking or missing considerations. Your critique should be structured and provide clear feedback on how the reasoning process can be improved or corrected. You must think critically and provide a well-reasoned critique of the reasoning process. But, in case, you agree with the reasoning process, you must provide a detailed explanation of why you agree with it. 

Show your process of thinking, in multiple reasoning steps and critique the reasoning process of the reasoning ai assistant or agree with it incase it seems sound. Develop multiple(minimum 10 and upto 25) thoughts, critique them, build on them, anticipate more reasonings to explore and then come to a final conclusion about the reasoning process of the reasoning ai assistant. You must intelligently ensure you do your job of finding fallacies or agreeing with the other ai assistant's reasoning.

Note: Ensure a very intellectually sound work, because, if not, then i may be fired from my job.
"""

critic_prompt = f"""
This is the reasoning process of the reasoning ai assistant:
{response.content}

Ensure to now reason and critic/agree with the assistant after building up your reasoning base thoughts and then steering towards what reasoning based you have, towards a decision of critiquing or agreeing with the reasoning process of the reasoning ai assistant. Do not dissatisfy me and ensure good working.
"""

# print("Invoking critic.....")
# response = model.invoke(
#     [
#         SystemMessage(content = sys_prompt_critic),
#         HumanMessage(content = critic_prompt)
#     ]
# )

# print("Response from llm:")
# print(response.content)
