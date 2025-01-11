# Import required libraries for file handling and hashing
import hashlib
import os
import tempfile
from pathlib import Path
import json

# Import libraries for data processing
import htmltabletomd
import unstructured_client
from diskcache import Cache
from unstructured_client.models import operations, shared
from config import IndexerConfig as Config
import logging


# Get the logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("parser.log", mode="a"))


class FileParser:
    """Class for parsing files into text format and handling subdocuments

    Args:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments

    Returns:
        FileParser: Instance of FileParser class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = Cache(".cache/parser-cache")
        self.config = Config()
        self.unstructured_client = unstructured_client.UnstructuredClient(
            api_key_auth=self.config.UNSTRUCTURED_API_KEY,
            server_url=self.config.UNSTRUCTURED_SERVER_URL,
        )

    def _get_cache_key(self, file_name: str, byte_data: bytes) -> str:
        """Generate a unique cache key based on file name and content

        Args:
            file_name (str): Name of the file
            byte_data (bytes): Binary content of the file

        Returns:
            str: Unique cache key
        """
        content_hash = hashlib.md5(byte_data).hexdigest()
        return f"{file_name}_{content_hash}"

    def obj_to_text(self, file_name: str, byte_data: bytes, **kwargs) -> str:
        """Convert file object to text using unstructured API

        Args:
            file_name (str): Name of the file
            byte_data (bytes): Binary content of the file

        Returns:
            str: Extracted text from the file
        """
        # Strip quotes if present
        file_name = file_name.strip('"')

        # Check cache
        cache_key = self._get_cache_key(file_name, byte_data)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key: {cache_key}")
            if cached_result == "":
                logger.info(f"Cache hit for key: {cache_key} but empty")
            else:
                return cached_result

        logger.info(f"Processing new text for key: {cache_key}")

        # Use tempfile for safer temporary file handling
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(byte_data)
            temp_file_path = temp_file.name

        try:
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=open(temp_file_path, "rb"),
                        file_name=file_name,
                    ),
                    strategy=shared.Strategy.HI_RES,
                    chunking_strategy=(
                        shared.ChunkingStrategy.BY_PAGE
                        if file_name.endswith(".pdf")
                        else shared.ChunkingStrategy.BASIC
                    ),
                    max_characters=6000 if file_name.endswith(".pdf") else None,
                    new_after_n_chars=5000 if file_name.endswith(".pdf") else None,
                    languages=["eng"],
                    split_pdf_page=True,
                    split_pdf_allow_failed=True,
                    split_pdf_concurrency_level=15,
                ),
            )

            res = self.unstructured_client.general.partition(request=req)

            # Process the response
            elements = []
            for element in res.elements:
                if element["type"] == "table":
                    elements.append(
                        htmltabletomd.convert_table(element["metadata"]["text_as_html"])
                    )
                else:
                    elements.append(element["text"])

            result_text = "\n".join(elements)
            self.cache.set(cache_key, result_text)
            return result_text

        except Exception as e:
            logger.error(f"Error processing document {file_name}: {e}")
            raise
        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)

    def parse_to_byte_array(self, byte_string: bytes) -> bytearray:
        """
        Converts a string representation of bytes (e.g., b'...') into a byte array.

        Args:
            byte_string (bytes): Input byte string.

        Returns:
            bytearray: Byte array representation.
        """
        return bytearray(byte_string)
