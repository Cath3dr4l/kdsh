# Import standard libraries
import hashlib
import json
import logging
import os
import sys
import anthropic
import diskcache
import pathway as pw
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from tqdm import tqdm

from config import IndexerConfig as Config

config = Config()


class CustomSplitter(pw.UDF):
    """A text splitter that adds contextual information to each chunk using Claude.

    Args:
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        separators (list[str]): List of separators to use for splitting
        api_key (str): Anthropic API key
        cache_dir (str): Directory to store the disk cache
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = None,
        api_key: str = None,
        cache_dir: str = ".cache/contextual-splitter",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

        # Get the logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("splitter.log", mode="a"))

    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:
        """Split the text into chunks using LangChain's splitter and add file context."""
        self.logger.info("Starting text splitting process.")
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)

        # Update splitter if parameters changed
        if chunk_size != self.chunk_size or chunk_overlap != self.chunk_overlap:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.separators,
            )

        # Process each chunk with document context
        original_chunks = self.splitter.split_text(txt)

        self.logger.info(
            f"Split text into {len(original_chunks)} chunks of size {chunk_size}."
        )
        return [(chunk, {}) for chunk in original_chunks]

        self.logger.info("Text splitting process completed.")

    # Public interface for splitting
    def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Split given text into overlapping chunks with file context."""
        return super().__call__(text, **kwargs)
