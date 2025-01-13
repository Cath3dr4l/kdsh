from dotenv import load_dotenv

load_dotenv()
import sys
import os
from typing import List, Dict, Any, Optional, Set
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from llm_based_classifier import LLMBasedClassifier
from rag_based_classifier import RagBasedClassifier
import asyncio


class FinalClassifier:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3)

        self.llm_classifier = LLMBasedClassifier()
        self.rag_classifier = RagBasedClassifier()

    async def get_title_and_abstract(self, paper_content):
        sys_prompt = """
          You are expert in extracting the title and abstract from a research paper. You need to extract the title and abstract from the given paper content.
            Format your output exactly as follows:
                <title> [Title of the paper] </title>
                <abstract> [Abstract of the paper] </abstract>
            For example:
                <title> Highway Graph to Accelerate Reinforcement Learning </title>
                <abstract> 
                    Reinforcement Learning (RL) algorithms often struggle with low training efficiency. A common approach to address this challenge is integrating model-based planning algorithms, such
                    as Monte Carlo Tree Search (MCTS) or Value Iteration (VI), into the environmental model.
                    However, VI faces a significant limitation: it requires iterating over a large tensor with dimensions |S| × |A| × |S|, where S and A represent the state and action spaces, respectively.
                    This process updates the value of the preceding state st−1 based on the succeeding state st
                    through value propagation, resulting in computationally intensive operations. To enhance
                    the training efficiency of RL algorithms, we propose improving the efficiency of the value
                    learning process. In deterministic environments with discrete state and action spaces, we
                    observe that on the sampled empirical state-transition graph, a non-branching sequence of
                    transitions—termed a highway—can take the agent directly from s0 to sT without deviation
                    through intermediate states. On these non-branching highways, the value-updating process
                    can be streamlined into a single-step operation, eliminating the need for iterative, step-bystep updates. Building on this observation, we introduce a novel graph structure called
                    the highway graph to model state transitions. The highway graph compresses the transition
                    model into a compact representation, where edges can encapsulate multiple state transitions,
                    enabling value propagation across multiple time steps in a single iteration. By integrating
                    the highway graph into RL (as a model-based off-policy RL method), the training process
                    is significantly accelerated, particularly in the early stages of training. Experiments across
                    four categories of environments demonstrate that our method learns significantly faster than
                    established and state-of-the-art model-free and model-based RL algorithms (often by a factor of 10 to 150) while maintaining equal or superior expected returns. Furthermore, a deep
                    neural network-based agent trained using the highway graph exhibits improved generalization capabilities and reduced storage costs. T
                </abstract>
        """

        user_prompt = f"<paper_content> {paper_content} </paper_content>"

        response = await self.model.ainvoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        return response.content

    async def classify(self, paper_content):
        # paper_content = await self.get_title_and_abstract(paper_content)
        # print(paper_content)
        llm_task = asyncio.create_task(self.llm_classifier.classify(paper_content))
        rag_task = asyncio.create_task(self.rag_classifier.classify(paper_content))

        llm_classifier_response = await llm_task
        rag_classifier_response = await rag_task

        sys_prompt = """
        You are an expert at classifying papers into one of five conferences. You need to classify into TMLR, NeurIPS, KDD, CVPR, EMNLP.
        you have response from LLMBasedClassifier and RagBasedClassifier, you need to compare and give final response.
        Format your output exactly as follows:
            <conference> [Conference Name] </conference>
            <rationale> [Provide a detailed explanation of your reasoning here, It should be around 100 words ] </rationale>

        For example:
            <conference> CVPR </conference>
            <rationale> The paper primarily focuses on computer vision techniques using LiDAR data, which aligns with the topics covered by CVPR. The use of clustering methods for object detection in 3D space further supports this classification. </rationale>
        """

        user_prompt = f"""<LLMBasedClassifier> {llm_classifier_response} </LLMBasedClassifier>
                        <RagBasedClassifier> {rag_classifier_response} </RagBasedClassifier>
                        Now, based on the responses from both classifiers, provide the final classification for this paper
                        """

        response = await self.model.ainvoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        return response.content


if __name__ == "__main__":
    import asyncio
    import PyPDF2

    classifier = FinalClassifier()
    pdf_dir = "../dataset/Papers"

    # List to store the content of the first 50 PDFs
    pdf_contents = []

    # Iterate over the PDF files named P001.pdf to P050.pdf in the directory
    for i in range(1, 2):
        pdf_file = f"P{i:03}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if os.path.exists(pdf_path):
            print(f"Processing: {pdf_path}")
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text()
                pdf_contents.append({"file": pdf_file, "content": content})

    content = pdf_contents[0]["content"]
    result = asyncio.run(classifier.classify(content))
    print(result)
