import asyncio

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import copy
import requests
import copy
from glob import glob
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from utils.get_similarity import get_similarity
from utils.embedding_client import EmbeddingClient
from utils.prompts import query_generation_sys_prompt, paper_summarizer_sys_prompt
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import aiohttp
from typing import List

load_dotenv()

from openai import AsyncOpenAI

client = AsyncOpenAI()


async def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = await client.embeddings.create(input=[text], model=model)
    return res.data[0].embedding


async def get_embedding_many(texts: List[str], model="text-embedding-3-small"):
    texts = [text.replace("\n", " ") for text in texts]
    res = await client.embeddings.create(input=texts, model=model)
    return res.data


class SimilarityBasedClassifier:
    def __init__(self):
        self.client = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, max_tokens=1000
        )  # Using LangChain's ChatOpenAI

        self.venue_conf_map = {
            "Trans. Mach. Learn. Res.": "TMLR",
            "Neural Information Processing Systems": "NeurIPS",
            "Knowledge Discovery and Data Mining": "KDD",
            "Conference on Empirical Methods in Natural Language Processing": "EMNLP",
            "Computer Vision and Pattern Recognition": "CVPR",
        }
        self.embedder = EmbeddingClient()

    def format_json(self, d):
        return (
            json.dumps(d, indent=4).replace("\n        ", " ").replace("\n    ]", " ]")
        )

    def filter_response(self, response):
        st_q_sec = response.find("<queries>")
        end_q_sec = response.find("</queries>")
        section = response[st_q_sec + 9 : end_q_sec]
        queries = []

        while "<query>" in section:
            st_q = section.find("<query>")
            end_q = section.find("</query>")
            query = section[st_q + 7 : end_q]
            queries.append(query.strip().lower())
            section = section[end_q + 8 :]
        return queries

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        user_content = "# The content of the paper is as follows:\n\n\n"
        for i, page in enumerate(reader.pages):
            page_content = f"## Page {i + 1}:\n{page.extract_text()}\n{'-' * 100}\n"
            user_content += page_content
        return user_content

    async def get_paper_summary(self, user_content):
        completion = await self.client.ainvoke(
            [
                {"role": "system", "content": paper_summarizer_sys_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return completion.content

    async def get_queries(self, user_content):
        response = await self.client.ainvoke(
            [
                {"role": "system", "content": query_generation_sys_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        return self.filter_response(response.content)

    async def get_unique_queries(self, queries):
        response = await self.client.ainvoke(
            [
                {
                    "role": "system",
                    "content": """
                 You are given a list of different queries Identify unique ones 
                 The queries section is the part which contains the final queries in the structured XML format.
    
        The output format for the whole queries' section is XML.
        Each query needs to be enclosed in XML tags: <query> and </query>.
        The queries section as whole needs to be enclosed in XML tags: <queries> and </queries>.
    
        ## Output format -
        ```
        # Thinking Section
        {$thinking_and_reasoning_for query_creation}
    
        # Queries section
        <queries>
            <query>{$query_1}</query>
            <query>{$query_2}</query>
            ...
            <query>{$query_n}</query>
        </queries>
        ```
                 """,
                },
                {"role": "user", "content": queries},
            ],
        )

        return self.filter_response(response.content)

    async def fetch_papers(self, query):

        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?"
            f"query={query.replace(' ', '+')}&limit=100&fields=title,url,abstract,venue,embedding.specter_v2&"
            "venue=Trans.+Mach.+Learn.+Res.,Neural+Information+Processing+Systems,"
            "Knowledge+Discovery+and+Data+Mining,Conference+on+Empirical+Methods+in+Natural+Language+Processing,"
            "Computer+Vision+and+Pattern+Recognition"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    print(f"Request failed with status code {response.status}")
                    print("Response Text:", await response.text())
                    return []

    async def process_query(self, query, summary_vector, venues_sim, venues_sim_empty):
        papers = await self.fetch_papers(query)
        query_venues_sim = copy.deepcopy(venues_sim_empty)
        papers_with_abstracts = []
        abstracts = []
        for paper in papers:
            if (
                "abstract" in paper
                and paper["abstract"] is not None
                and len(paper["abstract"]) > 10
            ):
                abstract = paper["abstract"]
                papers_with_abstracts.append(paper)
                abstracts.append(abstract)
            else:
                print("No abstract found in paper")
        if len(abstracts) > 0:
            # Get embeddings for all abstracts after collecting them
            abstract_vectors = await get_embedding_many(texts=abstracts)
            abstract_vectors = [
                abstract_vector.embedding for abstract_vector in abstract_vectors
            ]

        for idx, paper in enumerate(papers_with_abstracts):
            if True:
                abstract_vector = abstract_vectors[idx]
                similarity = get_similarity(abstract_vector, summary_vector)

                # Update similarity scores for both venues
                venues_sim[paper["venue"]][1] += similarity
                query_venues_sim[paper["venue"]][1] += similarity

                # Update the number of papers processed for the venue
                venues_sim[paper["venue"]][0] += 1
                query_venues_sim[paper["venue"]][0] += 1

                # Recalculate the average similarity score for the venue
                venues_sim[paper["venue"]][2] = (
                    venues_sim[paper["venue"]][1] / venues_sim[paper["venue"]][0]
                )
                query_venues_sim[paper["venue"]][2] = (
                    query_venues_sim[paper["venue"]][1]
                    / query_venues_sim[paper["venue"]][0]
                )

        # Sort venues by their average similarity score in descending order
        query_venues_sim = dict(
            sorted(query_venues_sim.items(), key=lambda item: item[1][2], reverse=True)
        )

        return query_venues_sim

    async def classify(self, user_content: str):
        venues_sim = {
            "Trans. Mach. Learn. Res.": [0, 0, 0],
            "Neural Information Processing Systems": [0, 0, 0],
            "Knowledge Discovery and Data Mining": [0, 0, 0],
            "Conference on Empirical Methods in Natural Language Processing": [
                0,
                0,
                0,
            ],
            "Computer Vision and Pattern Recognition": [0, 0, 0],
        }

        venues_sim_empty = copy.deepcopy(venues_sim)

        summary = await self.get_paper_summary(user_content)
        print("-" * 100)
        print("PAPER SUMMARY:\n\n")
        print(f"{summary}\n\n")

        queries_list = await asyncio.gather(
            *[self.get_queries(user_content) for _ in range(3)]
        )

        queries = await self.get_unique_queries(str(queries_list))

        async def process_iteration():

            print("-" * 100)
            print("# Queries Obtained: \n")
            for i, query in enumerate(queries):
                print(f"{i+1}. {query}")
            print("-" * 100)
            summary_vector = await get_embedding(summary)

            tasks = [
                self.process_query(query, summary_vector, venues_sim, venues_sim_empty)
                for query in queries
            ]
            results = await asyncio.gather(*tasks)

            for result in results:
                formatted_str = self.format_json(result)
                print(formatted_str)
                print("\n")

        await process_iteration()
        # await asyncio.gather(*[process_iteration() for _ in range(3)])

        venues_sim = dict(
            sorted(venues_sim.items(), key=lambda item: item[1][2], reverse=True)
        )
        formatted_str = self.format_json(venues_sim)
        print(formatted_str)

        result = ""
        for venue, data in venues_sim.items():
            result += f"""
                {venue}: 
                    No of papers found: {data[0]}
                    Average Similarity: {data[2]}
            """
            result += "\n"

        print(result)

        selected_venue = list(venues_sim.keys())[0]
        selected_conf = self.venue_conf_map[selected_venue]

        print()
        print("Conference Selected:", selected_conf)
        print("\n\n")
        return result


confs = ["CVPR", "EMNLP", "KDD", "NeurIPS", "TMLR"]


if __name__ == "__main__":
    dir_path = "../dataset/Reference/Publishable/"
    confs = ["CVPR", "EMNLP", "KDD", "NeurIPS", "TMLR"]
    import asyncio
    import PyPDF2
    import time

    pdf_dir = "../dataset/Reference/Publishable"
    classifier = SimilarityBasedClassifier()

    results = []
    for conf in confs[3:4]:

        dir_path = f"{pdf_dir}/"
        # Specify the PDF file path
        folder_path = f"{dir_path}{conf}/"
        print(f"Getting files for conference: {conf}\n")
        files = glob(f"{folder_path}*")

        for file in files:
            pdf_path = file
            start = time.time()
            print(f"Checking for file: {pdf_path}")

            # Open and read the PDF
            reader = PyPDF2.PdfReader(pdf_path)

            user_content = "# The content of the paper is as follows:\n\n\n"

            # Extract text from each page
            for i, page in enumerate(reader.pages):
                page_content = f"## Page {i + 1}:\n{page.extract_text()}\n{'-' * 100}\n"
                user_content += page_content

            result = asyncio.run(classifier.classify(user_content))
            end = time.time()
            new_res = {
                "file": file,
                "true conferece": conf,
                "suggested conference": result,
                "time taken": end - start,
                # "rationale": result["rationale"],
            }
            print(new_res)
            print("\n\n\n")
            results.append(new_res)

    # pdf_dir = "../dataset/Papers"

    # # List to store the content of the first 50 PDFs
    # pdf_contents = []

    # # measure time
    # start = time.time()

    # # Iterate over the PDF files named P001.pdf to P050.pdf in the directory
    # for i in range(1, 2):
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

    # content = pdf_contents[0]["content"]

    # classifier = SimilarityBasedClassifier()
    # res = asyncio.run(classifier.classify(content))

    # # measure time
    # end = time.time()
    # print("Time taken to classify the paper:", end - start)
    # print(res)
