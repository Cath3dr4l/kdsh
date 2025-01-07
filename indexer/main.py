from config import IndexerConfig
from services.indexer import Indexer
import pathway as pw


if __name__ == "__main__":
    config = IndexerConfig()

    indexer = Indexer(config)
    pw.run_all(terminate_on_error=False)
