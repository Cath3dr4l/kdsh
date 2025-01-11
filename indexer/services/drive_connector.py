# Import standard libraries
import os
import re
import sys
from pathway import Json

# Import pathway library for data processing
import pathway as pw
import logging

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom utility modules
from utils.file_parser import FileParser

import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

conference_to_id_map = {
    "1VU_Np53I7HtC-t3IAzTZQzodXXTgM4cA": "TMLR",
    "13EITinaXo5Bw06R66HE_CuHohi7lzoWe": "NeurIPS",
    "1sPkEvh-13LJtzoxGKIcpukDdySLiVkYa": "KDD",
    "1aslYCqHnOpBTGt6YB0mZJ7wEKJayfIbB": "EMNLP",
    "1npEk3caORb-tWIdx22EoZ7fxiAPgrem6": "CVPR",
}


class DriveConnector:
    """
    Class responsible for interacting with Google Drive, fetching, and processing file content.

    Args:
        folder_id (str): The ID of the Google Drive folder to connect to
        mode (str, optional): Connection mode. Defaults to "streaming"
        refresh_interval (int, optional): Refresh interval in seconds. Defaults to 30

    Returns:
        None
    """

    def __init__(
        self, folder_id: str, mode: str = "streaming", refresh_interval: int = 30
    ):
        self.folder_id = folder_id
        self.table = self._drive_connector(
            folder_id=folder_id, mode=mode, refresh_interval=refresh_interval
        )

    def _drive_connector(
        self,
        folder_id: str,
        mode: str = "streaming",
        refresh_interval: int = 30,
    ):
        """
        Streams file content from Google Drive based on the folder ID.

        Args:
            folder_id (str): Google Drive folder ID.
            mode (str): Connection mode
            refresh_interval (int): Refresh interval in seconds

        Returns:
            pw.Table: The table containing file content from Google Drive.
        """

        table = pw.io.gdrive.read(
            object_id=folder_id,
            service_user_credentials_file=os.path.join(
                os.path.dirname(__file__), "credentials.json"
            ),
            mode=mode,
            refresh_interval=refresh_interval,
            with_metadata=True,
        )
        return table

    def get_parsed_table(self):
        """
        Processes the file data by extracting its type and content.

        Args:
            table (pw.Table): Table containing file metadata and data.

        Returns:
            pw.Table: Table with processed file content including metadata and parsed data.
        """

        # UDF to extract and process file content with metadata
        @pw.udf
        def get_file_content(metadata, data):
            file_parser = FileParser()
            byte_data = file_parser.parse_to_byte_array(data)

            file_name = str(metadata["name"])

            existing_metadata = str(metadata)
            existing_metadata = json.loads(existing_metadata)

            extracted_text = file_parser.obj_to_text(
                file_name=file_name, byte_data=byte_data
            ).encode("utf-8")

            return extracted_text

        # New UDF to modify metadata and add conference information
        @pw.udf
        def modify_metadata(metadata):
            existing_metadata = str(metadata)
            metadata = json.loads(existing_metadata)
            conference_id = metadata.get("parents")[0]
            if conference_id in conference_to_id_map:
                metadata["conference"] = conference_to_id_map[conference_id]
            return metadata

        # Add processed data column to table
        result_table = self.table.with_columns(
            data=get_file_content(self.table._metadata, self.table.data),
        )

        # Modify metadata to include conference information
        result_table = result_table.with_columns(
            _metadata=modify_metadata(self.table._metadata)
        )

        return result_table
