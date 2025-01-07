# Import required libraries
import os
import sys
import logging
import hashlib
from config import IndexerConfig as Config
import pprint
import json
from diskcache import Cache

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anthropic
from openai import OpenAI
from collections import Counter

# Updated sample JSON structure for research papers
sample_json = {
    "document_type": "research paper",
    "authors": ["john smith", "sarah jones"],
    "research_fields": ["machine learning", "computer vision"],
    "publication_year": 2023,
    "keywords": ["deep learning", "neural networks", "image classification"],
    "institutions": ["stanford university", "mit"],
}

# Updated prompt template for research papers
prompt = """
Analyze the provided research document and extract structured insights with a focus on academic and scientific content.

<document_name>
{doc_name}
</document_name>

<document>
{doc_text}
</document>

Return the results as a JSON object with the following fields, ensuring type safety and using lowercase for all values:

- "document_type" (string): The specific type of academic document (e.g., "research paper", "conference paper", "thesis", "review article", "technical report").
- "authors" (list of strings): Names of the authors of the paper.
- "research_fields" (list of strings): Primary research areas or disciplines covered in the paper.
- "publication_year" (integer): The year the paper was published.
- "keywords" (list of strings): Key technical terms and concepts from the paper.
- "institutions" (list of strings): Academic or research institutions affiliated with the authors.

Ensure all text values in the JSON output are in lowercase.

Example output:
{sample_json}

Return only the JSON object in the specified format.
"""


class MetadataEnhancer:
    """
    A class to enhance document metadata using AI models (OpenAI or Claude).

    Args:
        model_type (str): Type of AI model to use ('openai' or 'claude')
    """

    def __init__(self, model_type="openai"):
        self.config = Config()
        if model_type == "claude":
            self.client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
            self.model = "claude-3-haiku-20240307"
        elif model_type == "openai":
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            self.model = "gpt-4o"
        else:
            raise ValueError("Invalid model type. Choose 'claude' or 'openai'.")

        # Initialize logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

        # Initialize diskcache
        self.cache = Cache(".cache/metadata_cache")

    def generate_cache_key(self, doc_text, doc_name):
        """
        Generate a hash-based cache key from the document text and name.

        Args:
            doc_text (str): Text content of the document
            doc_name (str): Name of the document

        Returns:
            str: SHA-256 hash of the combined document text and name
        """
        key_data = f"{doc_name}:{doc_text[:1000]}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def enhance(self, doc_text, doc_name):
        """
        Enhance the metadata using AI to extract research paper insights.

        Args:
            doc_text (str): Text content of the document
            doc_name (str): Name of the document

        Returns:
            dict: Enhanced metadata with document_type, authors, research_fields, etc.
        """
        cache_key = self.generate_cache_key(doc_text, doc_name)

        # Check if the result is already in the cache
        if (
            cache_key in self.cache
            and self.cache[cache_key] is not None
            and self.cache[cache_key]["document_type"] is not None
        ):
            self.logger.info(f"Retrieved enhanced metadata from cache for {doc_name}")
            return self.cache[cache_key]

        content = prompt.format(
            doc_text=doc_text,
            doc_name=doc_name,
            sample_json=json.dumps(sample_json, indent=4),
        )

        try:
            if self.model == "gpt-4o":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=1000,
                    temperature=0.0,
                )
                enhanced_data = response.choices[0].message.content
                enhanced_data = enhanced_data.replace("```json", "").replace("```", "")
            else:
                response = self.client.beta.prompt_caching.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                )
                enhanced_data = response.content[0].text

            self.logger.info(f"Response usage: {response.usage}")

            # Merge the original metadata with the enhanced data
            addn_metadata = json.loads(enhanced_data)

            # Store the enhanced metadata in the cache
            self.cache[cache_key] = addn_metadata
            self.logger.info(f"Stored enhanced metadata in cache for {doc_name}")

            return addn_metadata

        except Exception as e:
            self.logger.error(f"Error enhancing metadata: {str(e)}")
            return {
                "document_type": None,
                "authors": [],
                "research_fields": [],
                "publication_year": None,
                "keywords": [],
                "institutions": [],
            }


class MetadataEnhancerForDoc(MetadataEnhancer):
    """
    A specialized metadata enhancer for processing multiple sub-documents.
    Aggregates metadata from multiple sections of a document.
    """

    def __init__(self):
        super().__init__()

    def enhance(self, sub_docs, doc_name):
        """
        Enhance metadata for multiple sub-documents and aggregate results.

        Args:
            sub_docs (list): List of document sections to process
            doc_name (str): Name of the document

        Returns:
            dict: Aggregated metadata with combined authors, fields, keywords, etc.
        """
        authors = set()
        research_fields = set()
        keywords = set()
        institutions = set()
        years = []
        document_types = []

        for sub_doc in sub_docs:
            metadata = super().enhance(sub_doc, doc_name)
            authors.update(metadata["authors"])
            research_fields.update(metadata["research_fields"])
            keywords.update(metadata["keywords"])
            institutions.update(metadata["institutions"])
            years.append(metadata["publication_year"])
            document_types.append(metadata["document_type"])

        counter = Counter(years)
        year = counter.most_common(1)[0][0]

        document_type_counts = Counter(document_types)
        document_type = document_type_counts.most_common(1)[0][0]

        return {
            "authors": list(authors),
            "research_fields": list(research_fields),
            "keywords": list(keywords),
            "institutions": list(institutions),
            "publication_year": year,
            "document_type": document_type,
        }


if __name__ == "__main__":
    # read the file test.txt
    with open("./test.txt", "r") as file:
        doc_text = file.read()
        doc_name = "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"
        metadata_enhancer = MetadataEnhancer()
        metadata = metadata_enhancer.enhance(doc_text, doc_name)
        pprint.pp(metadata)
