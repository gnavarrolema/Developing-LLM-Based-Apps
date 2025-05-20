from typing import List, Optional, Iterator
import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm

from backend.config import settings


class ETLProcessor:
    """
    This class is responsible for performing an Extract-Transform-Load (ETL)
    process for document embedding.
    """

    def __init__(
        self,
        batch_size: int,
        chunk_size: int,
        chunk_overlap: int,
        dataset_path: Optional[str] = settings.DATASET_PATH,
        embedding_model: Optional[str] = settings.EMBEDDINGS_MODEL,
        collection_name: Optional[str] = settings.CHROMA_COLLECTION,
        persist_directory: Optional[str] = settings.CHROMA_DB_PATH,
    ):
        """
        Initializes the ETLProcessor object with a specified batch_size.

        Parameters
        ----------
        batch_size : int
            Number of documents to process in each batch, e.g. 100.

        chunk_size : int
            Size of chunks to be split into, e.g. 500.

        chunk_overlap : int
            Number of characters to overlap between chunks, e.g. 20.

        dataset_path : str, optional
            Path to csv file containing the dataset.

        embedding_model : str, optional
            Name of the embedding model to be used, e.g.
            "paraphrase-MiniLM-L6-v2".

        collection_name : str, optional
            Name of the collection to be used in the vector store.

        persist_directory : str, optional
            Directory to persist the vector store.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.embedding = SentenceTransformerEmbeddings(
            model_name=embedding_model
        )
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len, 
            add_start_index=True
        )

    def _validate_dataset_exists(self):
        """Validates that the dataset file exists."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}"
            )

    def _validate_dataset_columns(self, df):
        """Validates that the dataset has all required columns."""
        required_columns = ["description", "Employment type", "Seniority level", 
                          "company", "location", "post_url", "title"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset CSV is missing required columns: {', '.join(missing_columns)}")
        return df[required_columns]

    def load_data_in_batches(self, chunk_size=10000) -> Iterator[pd.DataFrame]:
        """
        Loads the jobs descriptions from a CSV file in batches using pandas read_csv.
        
        Parameters
        ----------
        chunk_size : int
            Number of rows to load at once
            
        Yields
        ------
        pd.DataFrame
            Batch of job descriptions with required metadata
        """
        self._validate_dataset_exists()
        
        # Use pandas read_csv with chunksize parameter to process in batches
        for chunk in pd.read_csv(self.dataset_path, chunksize=chunk_size):
            # Validate and process each chunk
            try:
                validated_chunk = self._validate_dataset_columns(chunk)
                validated_chunk.dropna(subset=validated_chunk.columns, inplace=True)
                if not validated_chunk.empty:
                    yield validated_chunk
            except ValueError as e:
                import logging
                logging.error(f"Error processing CSV chunk: {e}")
                continue

    def create_documents(self, descriptions: pd.DataFrame) -> List[Document]:
        """
        Creates a list of Document objects from given descriptions.

        Parameters
        ----------
        descriptions : pd.DataFrame
            Job descriptions to be converted to Document objects.

        Returns
        -------
        List[Document]
            List of Document (langchain.docstore.document.Document) objects.
        """
        output_documents = []
        for idx, row in descriptions.iterrows():
            try:
                metadata = {
                    "employment_type": row["Employment type"],
                    "seniority_level": row["Seniority level"],
                    "company": row["company"],
                    "location": row["location"],
                    "post_url": row["post_url"],
                    "title": row["title"],
                    "id": idx,
                }
                doc = Document(page_content=row["description"], metadata=metadata)
                output_documents.append(doc)
            except Exception as e:
                import logging
                logging.error(f"Error creating document for index {idx}: {e}")
                continue
                
        return output_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller chunks using the pre-defined
        text_splitter.

        Parameters
        ----------
        documents : List[Document]
            List of Document objects to be split.

        Returns
        -------
        List[Document]
            List of split Document objects.
        """
        if not documents:
            return []
            
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            import logging
            logging.error(f"Error splitting documents: {e}")
            return []

    def process_batches(self, splits: List[Document]) -> None:
        """
        Processes documents in batches, adding them to a single Chroma instance.

        Parameters
        ----------
        splits : List[Document]
            List of Document objects to be processed.

        Returns
        -------
        None
        """
        if not splits:
            print("No documents to process.")
            return
            
        # Create a single Chroma instance
        vectordb = Chroma(
            embedding_function=self.embedding,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        # Process in batches
        for i in tqdm(
            range(0, len(splits), self.batch_size), desc="Processing batches"
        ):
            batch = splits[i: i + self.batch_size]
            if batch:  # Solo procesar si el lote no está vacío
                try:
                    # Add documents to the existing collection
                    vectordb.add_documents(documents=batch)
                except Exception as e:
                    import logging
                    logging.error(f"Error processing batch {i//self.batch_size}: {e}")
                    continue
            
        # Explicitly persist changes
        try:
            vectordb.persist()
            print(f"Successfully persisted {len(splits)} documents to the vector store.")
        except Exception as e:
            import logging
            logging.error(f"Error persisting vector store: {e}")

    def run_etl(self, limit=None) -> None:
        """
        Executes the ETL process: extract data from a source, transform it,
        and load into a new storage.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of documents to process. If None, process all documents.
        """
        total_processed = 0
        for batch_df in self.load_data_in_batches():
            if limit is not None and total_processed >= limit:
                break
                
            # Si hay un límite, ajustar el tamaño del lote actual
            if limit is not None:
                remaining = limit - total_processed
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]
            
            print(f"Processing batch of {len(batch_df)} documents...")
            docs = self.create_documents(batch_df)
            splits = self.split_documents(docs)
            self.process_batches(splits)
            
            total_processed += len(batch_df)
            if limit is not None:
                print(f"Processed {total_processed}/{limit} documents.")
            else:
                print(f"Processed {total_processed} documents so far.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ETL process for document embedding')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--chunk-size', type=int, default=500, help='Size of text chunks')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Overlap between chunks')
    
    args = parser.parse_args()
    
    etl_processor = ETLProcessor(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    etl_processor.run_etl(limit=args.limit)