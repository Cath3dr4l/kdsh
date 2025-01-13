from dotenv import load_dotenv

load_dotenv()
import sys
import os
from typing import List, Dict, Any, Optional, Set
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


class LLMBasedClassifier:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=3)

    def get_sys_prompt(self):
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
        return sys_prompt_actor

    def get_actor_prompt(self, content: str):
        actor_prompt = f"""
            I have a paper for you to analyze and categorize into one of the 5 conference categories you know.
            Think step by step, building up on your reasonings and ending up concluding with the mmost appropriate conference for the paper.
            Speak out your thinking and reasonings as you go through and think about various parts of the paper. Ensure thoughtts are rich, diverse, non-repetitive and after multiple thoughts(minimum 10 and upto 25 thoughts), you conclude withtthe final result as a one word answer which is the conference name. ENsure rich and critically thought out decisions
            You mmust crtique your previous thoughts, anticipate some futher reasonings to explore in later thoughts, contrast with other thoughts, and hence overall, have a cohesive and good thought process to come to a final well-reasoned conclusion.

            Here is the paper's content:
            {content}
          """
        return actor_prompt

    async def classify(self, content: str):
        sys_prompt_actor = self.get_sys_prompt()
        actor_prompt = self.get_actor_prompt(content)
        response = await self.model.ainvoke(
            [
                SystemMessage(content=sys_prompt_actor),
                HumanMessage(content=actor_prompt),
            ]
        )
        return response.content


if __name__ == "__main__":
    classifier = LLMBasedClassifier()
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
    result = classifier.classify(content)
    print(result)
