from pathlib import Path
from typing import Optional
import os

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(".env"))

root = Path(__file__).parent.parent

class Settings(BaseSettings):
    # NLP Models settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL: Optional[str] = os.getenv("OPENAI_LLM_MODEL")
    LANGCHAIN_VERBOSE: bool = False

    # Document Ingestion
    # Usar os.path.join() para rutas de archivos
    DATASET_PATH: Optional[str] = os.path.join(root, "dataset", "jobs.csv")
    CHROMA_DB_PATH: Optional[str] = os.path.join(root, "chroma")
    CHROMA_COLLECTION: Optional[str] = "jobs"
    EMBEDDINGS_MODEL: Optional[str] = "paraphrase-MiniLM-L6-v2"

    # Email settings
    SENDER_EMAIL_ADDRESS: Optional[str] = ""
    SENDER_EMAIL_PASSWORD: Optional[str] = ""

settings = Settings()