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

from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from services.agent import ModelConfig, LLMFactory, TreeOfThoughts, PaperEvaluator, ModelProvider


def main():
    # Configure models
    reasoning_config = ModelConfig(
        provider=ModelProvider.OPENAI, model_name="gpt-4o-mini"
    )

    critic_config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4o-mini")

    print("Configuring evaluator...")

    evaluator = PaperEvaluator(reasoning_config, critic_config)
    pdf_directory = "/home/divyansh/code/kdsh/dataset/Reference/Non-Publishable"
    pdf_path = "/home/divyansh/code/kdsh/dataset/Reference/Non-Publishable/R004.pdf"

    if os.path.exists(pdf_directory):
        print(f"Analyzing papers in {pdf_directory}...")
    else:
        print(f"Directory {pdf_directory} does not exist.")
        sys.exit(1)

    # for filename in os.listdir(pdf_directory):
    #     if filename.endswith(".pdf"):
    #         pdf_path = os.path.join(pdf_directory, filename)
    #         content = extract_pdf_content(pdf_path)

    #         decision = evaluator.evaluate_paper(content)

    #         print(f"\nResults for {filename}:")
    #         print(f"Publishable: {decision.is_publishable}")
    #         print("\nKey Strengths:")
    #         for strength in decision.primary_strengths:
    #             print(f"- {strength}")
    #         print("\nCritical Weaknesses:")
    #         for weakness in decision.critical_weaknesses:
    #             print(f"- {weakness}")
    #         print(f"\nRecommendation:\n{decision.recommendation}")

    content = extract_pdf_content(pdf_path)

    print(content)

    print("Content extracted from PDF")
    print("Analyzing paper...")

    decision = evaluator.evaluate_paper(content)

    print(f"\nResults:")
    print(f"Publishable: {decision.is_publishable}")
    print("\nKey Strengths:")
    for strength in decision.primary_strengths:
        print(f"- {strength}")
    print("\nCritical Weaknesses:")
    for weakness in decision.critical_weaknesses:
        print(f"- {weakness}")
    print(f"\nRecommendation:\n{decision.recommendation}")


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
            max_characters=2000,
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
            if filename.endswith(".pdf"):
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
                    if filename.endswith(".pdf"):
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
    paper_path: str, true_label: bool, evaluator: PaperEvaluator
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

        # Create result dictionary
        result = {
            "paper_id": os.path.basename(paper_path),
            "true_label": true_label,
            "predicted_label": decision.is_publishable,
            "conference": (
                os.path.basename(os.path.dirname(paper_path))
                if true_label
                else "Non-Publishable"
            ),
            "primary_strengths": "|".join(decision.primary_strengths),
            "critical_weaknesses": "|".join(decision.critical_weaknesses),
            "recommendation": decision.recommendation,
            "correct_prediction": true_label == decision.is_publishable,
            "file_path": paper_path,
        }

        return result

    except Exception as e:
        print(f"Error processing {paper_path}: {str(e)}")
        return {
            "paper_id": os.path.basename(paper_path),
            "true_label": true_label,
            "predicted_label": None,
            "conference": (
                os.path.basename(os.path.dirname(paper_path))
                if true_label
                else "Non-Publishable"
            ),
            "primary_strengths": "",
            "critical_weaknesses": "",
            "recommendation": f"ERROR: {str(e)}",
            "correct_prediction": False,
            "file_path": paper_path,
        }


def analyze_results(df: pd.DataFrame) -> None:
    """
    Analyze and print evaluation results.
    """
    total_papers = len(df)
    valid_predictions = df["predicted_label"].notna()
    df_valid = df[valid_predictions]

    if len(df_valid) == 0:
        print("\nNo valid predictions to analyze!")
        return

    correct_predictions = df_valid["correct_prediction"].sum()
    accuracy = (correct_predictions / len(df_valid)) * 100

    print("\nEvaluation Results Analysis:")
    print(f"Total papers processed: {total_papers}")
    print(f"Papers with valid predictions: {len(df_valid)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Confusion matrix for valid predictions
    true_pos = len(
        df_valid[
            (df_valid["true_label"] == True) & (df_valid["predicted_label"] == True)
        ]
    )
    true_neg = len(
        df_valid[
            (df_valid["true_label"] == False) & (df_valid["predicted_label"] == False)
        ]
    )
    false_pos = len(
        df_valid[
            (df_valid["true_label"] == False) & (df_valid["predicted_label"] == True)
        ]
    )
    false_neg = len(
        df_valid[
            (df_valid["true_label"] == True) & (df_valid["predicted_label"] == False)
        ]
    )

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


# def main():
#     # Configure models
#     reasoning_config = ModelConfig(
#         provider=ModelProvider.OPENAI,
#         model_name="gpt-4o-mini"
#     )

#     critic_config = ModelConfig(
#         provider=ModelProvider.OPENAI,
#         model_name="gpt-4o-mini"
#     )

#     print("Configuring evaluator...")
#     evaluator = PaperEvaluator(reasoning_config, critic_config)

#     # Base directory containing the papers
#     base_dir = "/home/divyansh/code/kdsh/dataset/Reference"

#     print(f"\nProcessing papers in {base_dir}...")

#     # Process all papers and get results DataFrame
#     results_df = process_papers_directory(base_dir, evaluator)

#     # Analyze and print results
#     analyze_results(results_df)

if __name__ == "__main__":
    main()
