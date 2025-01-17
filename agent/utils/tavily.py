import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

import re
import json
import PyPDF2

load_dotenv()


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


# Initialize Tavily Search tool
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,  # Set to True if images are needed
    api_key=os.getenv("TAVILY_API_KEY"),  # Load API key from environment variable
)


class Mosambi:
    def __init__(self):
        self.tavily = tool
        self.client = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, max_tokens=1000
        )  # Using LangChain's ChatOpenAI

    def retrieve(self, query: str):
        # Use Tavily Search to find relevant results
        search_results = self.tavily.run(query)
        return search_results

    def classify(self, paper_content):
        # Retrieve relevant content using Tavily
        retrieved_content = self.retrieve(paper_content)

        # Prepare the system prompt with retrieved content
        sys_prompt = """
      You are an expert in classifying academic papers into one of the following five conferences: TMLR, NeurIPS, KDD, CVPR, EMNLP. 
Your task is to classify the given paper into one of these conferences based on its content and the retrieved information. 

Below are some relevant search results about the paper that may provide additional context. Analyze the retrieved information carefully and determine the most appropriate conference for the paper. 
Provide a rationale that explains why the selected conference is the best fit for the paper, considering the topics, methods, and scope of the research.

Format your output exactly as follows:
<conference> [Conference Name] </conference>
<rationale> [Provide a detailed explanation of your reasoning here] </rationale>

For example:
<conference> CVPR </conference>
<rationale> The paper primarily focuses on computer vision techniques using LiDAR data, which aligns with the topics covered by CVPR. The use of clustering methods for object detection in 3D space further supports this classification. </rationale>

Here are the search results related to the paper: \n

        """

        sys_prompt += str(retrieved_content)

        user_prompt = f"""
        Now, based on the retrieved results and the content of the paper, classify this paper into one of the five conferences:
        {paper_content}
        """

        # Call LangChain's ChatOpenAI client with the system and user prompts
        response = self.client.invoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Return the model's response
        return output_to_json(response.content)


# if __name__ == "__main__":
#     mosambi = Mosambi()
#     content = """

# """
#     print(mosambi.classify(content))


if __name__ == "__main__":
    mosambi = Mosambi()

    # Directory containing the PDF files
    pdf_dir = "../dataset/Papers"

    # List to store the content of the first 50 PDFs
    pdf_contents = []

    import csv

    # Read the output.csv file to get the PDF names and publishable status
    csv_file = "output.csv"
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            pdf_file = f"{row['id']}.pdf"
            pdf_path = os.path.join(pdf_dir, pdf_file)
            if os.path.exists(pdf_path):
                print(f"Processing: {pdf_path}")
                with open(pdf_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    content = ""
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        content += page.extract_text()
                    pdf_contents.append(
                        {
                            "file": row["id"],
                            "content": content,
                            "publishable": row["publishable"],
                        }
                    )

    print(f"Total PDFs processed: {len(pdf_contents)}")

    # Classify the content of all the PDFs
    results = []
    for pdf in pdf_contents:
        if pdf["publishable"] == "True" or pdf["publishable"] == True:
            result = mosambi.classify(pdf["content"])
        else:
            result = {"conference": "na", "rationale": "na"}
        new_res = {
            "file": pdf["file"],
            "publishable": pdf["publishable"],
            "conference": result["conference"],
            "rationale": result["rationale"],
        }
        results.append(new_res)
        # Save the classification results to a csv file
        import pandas as pd

        results_df = pd.DataFrame(results)
        results_df.to_csv("classification_results.csv", index=False)

