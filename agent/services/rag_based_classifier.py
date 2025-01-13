import requests
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import aiohttp

load_dotenv()


async def retrieve(query: str, k: int):
    url = "http://0.0.0.0:8666/v1/retrieve"
    payload = {"query": query, "k": k}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            return await response.json()


class RagBasedClassifier:
    def __init__(self):
        self.retrieve = retrieve
        self.client = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, max_tokens=1000
        )  # Using LangChain's ChatOpenAI

    async def classify(self, paper_content):
        retrieved_content = await self.retrieve(paper_content, k=6)

        sys_prompt = """
        You are an expert at classifying papers into one of five conferences. You need to classify into TMLR, NeurIPS, KDD, CVPR, EMNLP.
        Here are some similar chunks present in a vectorstore with papers from these five conferences.
        
        """

        for conference in ["TMLR", "CVPR", "KDD", "EMNLP", "NeurIPS"]:
            current_chunks = ""
            for chunk in retrieved_content:
                if chunk["metadata"]["conference"] == conference:
                    current_chunks += chunk["text"]
                    current_chunks += "\n\n\n"
            if len(current_chunks) > 0:
                sys_prompt += "\n\n\n"
                sys_prompt += conference
                sys_prompt += current_chunks

        user_prompt = """
        Now, classify this paper according to the given guidelines.
        """

        user_prompt += paper_content

        # Call LangChain's ChatOpenAI client with the system and user prompt
        response = await self.client.ainvoke(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # return response["content"].
        return response.content


if __name__ == "__main__":
    mosambi = RagBasedClassifier()
    content = """
E!icient Mixture of Experts based on Large Language Models for
Low-Resource Data Preprocessing
Mengyi Yan
yanmy@act.buaa.edu.cn
Beihang University, Beijing, China
Yaoshu Wang∗
yaoshuw@sics.ac.cn
Shenzhen Institute of Computing
Sciences, Shenzhen, China
Kehan Pang
pangkehan@buaa.edu.cn
Beihang University, Beijing, China
Min Xie
xiemin@sics.ac.cn
Shenzhen Institute of Computing
Sciences, Shenzhen, China
Jianxin Li∗
lijx@buaa.edu.cn
Beihang University, Beijing, China
ABSTRACT
Data preprocessing (DP) that transforms erroneous and raw data to
a clean version is a cornerstone of the data mining pipeline. Due to
the diverse requirements of downstream tasks, data scientists and
domain experts have to handcraft domain-speci!c rules or train ML
models with annotated examples, which is costly/time-consuming.
In this paper, we present MELD (Mixture of Experts on Large Language Models for Data Preprocessing), a universal solver for lowresource DP. MELD adopts a Mixture-of-Experts (MoE) architecture
that enables the amalgamation and enhancement of domain-speci!c
experts trained on limited annotated examples. To!ne-tune MELD,
we develop a suite of expert-tuning and MoE-tuning techniques, including a retrieval augmented generation (RAG) system, meta-path
search for data augmentation, expert re!nement and router network training based on information bottleneck. To further verify the
e"ectiveness of MELD, we theoretically prove that MoE in MELD
is superior than a single expert and the router network is able to
dispatch data to the right experts. Finally, we conducted extensive
experiments on 19 datasets over 10 DP tasks to show that MELD
outperforms the state-of-the-art methods in both e"ectiveness and
e#ciency. More importantly, MELD is able to be! ne-tuned in a lowresource environment, e.g., a local, single and low-priced 3090 GPU.
The codes, datasets and full version of the paper are available [1].

    """
    print(mosambi.classify(content))
