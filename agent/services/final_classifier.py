from dotenv import load_dotenv

load_dotenv()
import sys
import os
from typing import List, Dict, Any, Optional, Set
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from services.llm_based_classifier import LLMBasedClassifier
from services.rag_based_classifier import RagBasedClassifier
from services.similarity_based_classifier import SimilarityBasedClassifier
import re
import asyncio
from models.schemas import ClassificationResponse, ClassifierPrediction


def output_to_json(output: str) -> dict:

    # Extract the conference and rationale using regex
    conference_match = re.search(
        r"<conference>\s*(.*?)\s*</conference>", output, re.IGNORECASE
    )
    rationale_match = re.search(
        r"<rationale>\s*(.*?)\s*</rationale>", output, re.IGNORECASE
    )

    # Get the matched text or set as None if not found
    conference = conference_match.group(1).strip() if conference_match else None
    rationale = rationale_match.group(1).strip() if rationale_match else None

    # Create a dictionary
    result = {"conference": conference, "rationale": rationale}

    return result


class FinalClassifier:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=3)

        self.llm_classifier = LLMBasedClassifier()
        self.rag_classifier = RagBasedClassifier()
        self.similarity_classifier = SimilarityBasedClassifier()

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

    async def summarize_thoughts(self, llm_classifier_response):
        sys_prompt = """
          You are an expert in summarizing thoughts and ideas. You need to summarize the classification suggestions provided by the classifiers. make it short and concise but do not miss out on any important information.
            Format your output exactly as follows:
                <summary> [Summary of the classification suggestions] </summary>
            For example:
                <summary> The classifiers suggest that the paper is best suited for the CVPR conference based on its focus on computer vision techniques and image processing. </summary>
        """

        user_prompt = f"<llm_classifier_response> {llm_classifier_response} </llm_classifier_response>"

        response = await self.model.ainvoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        return response.content

    async def classify(self, paper_content):

        llm_task = asyncio.create_task(self.llm_classifier.classify(paper_content))
        rag_task = asyncio.create_task(self.rag_classifier.classify(paper_content))
        similarity_task = asyncio.create_task(
            self.similarity_classifier.classify(paper_content)
        )
        llm_classifier_response = await llm_task
        rag_classifier_response = await rag_task
        similarity_classifier_response = await similarity_task

        llm_classifier_response = await self.summarize_thoughts(llm_classifier_response)

        sys_prompt = """
        You are a scientific paper classification expert specializing in categorizing computer science research papers into five major conferences: TMLR (Transactions on Machine Learning Research), NeurIPS (Neural Information Processing Systems), KDD (Knowledge Discovery and Data Mining), CVPR (Computer Vision and Pattern Recognition), and EMNLP (Empirical Methods in Natural Language Processing).

Input Format:
For each paper, you will receive classification suggestions from three different classifiers:
1. LLMBasedClassifier: Uses language model analysis
2. RagBasedClassifier: Uses retrieval-augmented generation
3. SimilarityBasedClassifier: Uses paper similarity metrics with historical conference papers

Required Output Format: 
<conference> [Conference Name] </conference>
<rationale> [A detailed 100-word explanation focusing on the paper's content, methodology, and fit with the conference's scope. The rationale should analyze the paper's core contributions, technical approach, and application domain.] </rationale>

Classification Guidelines:
- TMLR: Theoretical machine learning, algorithm development, learning theory
- NeurIPS: Deep learning, neural networks, AI systems, theoretical ML
- KDD: Data mining, large-scale data analysis, practical applications
- CVPR: Computer vision, image processing, visual understanding
- EMNLP: Natural language processing, computational linguistics

Important Notes:
1. The rationale must focus on the paper's content and its alignment with the conference themes
2. Avoid mentioning classifier votes or classification methods in the rationale
3. Base your decision on paper content, methodology, and domain fit
4. Consider each conference's unique focus and scope

Example:
<conference> CVPR </conference>
<rationale> This paper introduces a novel 3D object detection framework using LiDAR point clouds. The methodology combines deep learning with geometric clustering for robust real-time object recognition in autonomous driving scenarios. The focus on visual perception, spatial understanding, and real-world computer vision applications strongly aligns with CVPR's emphasis on advancing visual recognition systems and 3D scene understanding. </rationale> 
        """

        user_prompt = f"""<LLMBasedClassifier> {llm_classifier_response} </LLMBasedClassifier>
                        <RagBasedClassifier> {rag_classifier_response} </RagBasedClassifier>
                        <SimilarityBasedClassifier> {similarity_classifier_response} </SimilarityBasedClassifier>
                        
                        <paper_content> {paper_content} </paper_content>
                        Now, based on the responses from both classifiers, provide the final classification for this paper
                        """

        response = await self.model.ainvoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        print("Token count : ", response.response_metadata["token_usage"])

        return output_to_json(response.content)

    async def classify_with_details(self, paper_content: str) -> ClassificationResponse:
        # Get predictions from all classifiers in parallel
        llm_task = asyncio.create_task(self.llm_classifier.classify(paper_content))
        rag_task = asyncio.create_task(self.rag_classifier.classify(paper_content))
        similarity_task = asyncio.create_task(
            self.similarity_classifier.classify(paper_content)
        )

        # Await all results
        llm_result = await llm_task
        rag_result = await rag_task
        similarity_result = await similarity_task

        # Format individual predictions
        llm_prediction = ClassifierPrediction(
            conference=llm_result.conference,
            rationale=llm_result.rationale,
            thought_process=llm_result.thought_process,
        )

        rag_prediction = ClassifierPrediction(
            conference=self._extract_conference(rag_result),
            rationale=rag_result,  # Using full RAG output as rationale
        )

        similarity_prediction = ClassifierPrediction(
            conference=self._extract_conference(similarity_result),
            rationale=similarity_result,  # Using full similarity output as rationale
        )

        # Get final prediction using existing logic
        final_result = await self._get_final_prediction(
            paper_content, llm_result, rag_result, similarity_result
        )

        return ClassificationResponse(
            final_prediction=ClassifierPrediction(
                conference=final_result["conference"],
                rationale=final_result["rationale"],
            ),
            llm_prediction=llm_prediction,
            rag_prediction=rag_prediction,
            similarity_prediction=similarity_prediction,
            metadata={
                "token_usage": final_result.get("token_usage"),
                "processing_time": final_result.get("processing_time"),
            },
        )

    def _extract_conference(self, result: str) -> str:
        """Extract conference name from classifier output"""
        for conf in ["CVPR", "EMNLP", "KDD", "NeurIPS", "TMLR"]:
            if conf in result:
                return conf
        return "Unknown"

    async def _get_final_prediction(
        self, paper_content, llm_result, rag_result, similarity_result
    ):
        # Use existing classify method logic to get final prediction
        return await self.classify(paper_content)


confs = ["CVPR", "EMNLP", "KDD", "NeurIPS", "TMLR"]


if __name__ == "__main__":
    import asyncio
    import PyPDF2
    from glob import glob
    import time

    classifier = FinalClassifier()
    pdf_dir = "../dataset/Reference/Publishable"

    # List to store the content of the first 50 PDFs
    pdf_contents = []

    # Iterate over the PDF files named P001.pdf to P050.pdf in the directory
    # for i in range(1, 2):
    #     start = time.time()
    #     pdf_file = f"P{i:03}.pdf"
    #     pdf_path = os.path.join(pdf_dir, pdf_file)
    #     if os.path.exists(pdf_path):
    #         print(f"Processing: {pdf_path}")
    #         with open(pdf_path, "rb") as file:
    #             reader = PyPDF2.PdfReader(file)
    #             content = ""
    #             for page_num in range(len(reader.pages)):
    #                 page = reader.pages[page_num]
    #                 content += page.extract_text()
    #             pdf_contents.append({"file": pdf_file, "content": content})
    #     content = pdf_contents[0]["content"]
    #     result = asyncio.run(classifier.classify(content))
    #     end = time.time()
    #     print(f"Time taken: {end - start}")
    #     print(result)

    results = []
    for conf in confs[3:4]:
        dir_path = f"{pdf_dir}/"
        # Specify the PDF file path
        folder_path = f"{dir_path}{conf}/"
        print(f"Getting files for conference: {conf}\n")
        files = glob(f"{folder_path}*")

        for file in files:
            pdf_path = file
            print(f"Checking for file: {pdf_path}")

            # Open and read the PDF
            reader = PyPDF2.PdfReader(pdf_path)

            user_content = "# The content of the paper is as follows:\n\n\n"

            # Extract text from each page
            for i, page in enumerate(reader.pages):
                page_content = f"## Page {i + 1}:\n{page.extract_text()}\n{'-' * 100}\n"
                user_content += page_content

            result = asyncio.run(classifier.classify(user_content))
            new_res = {
                "file": file,
                "true conferece": conf,
                "suggested conference": result["conference"],
                "rationale": result["rationale"],
            }
            print(new_res)
            print("\n\n\n")
            results.append(new_res)

    import pandas as pd

    results_df = pd.DataFrame(results)
    results_df.to_csv("task2.1.csv", index=False)
