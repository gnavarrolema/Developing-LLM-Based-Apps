# Developing LLM-Based Apps

![image](https://github.com/user-attachments/assets/440f0c9f-427c-4068-927f-63ae7cd84ec1)


## Description

This project is designed to develop applications based on Large Language Models (LLM). It includes tools for data extraction, transformation, and loading, as well as the capability to interact with LLM-based assistants like ChatGPT and Jobs Finder Assistant.

## Features

-   **LLM-Based Assistants**: Interact with different assistants for various tasks.
-   **Document Processing**: Extract text from PDF files and process them for analysis.
-   **Similarity Search**: Use embedding models to perform similarity searches in a vector store.
-   **Easy Configuration**: Set up the project easily with a `.env` file.

## Contents

-   `backend/`
    -   `app.py`: Configuration and handling of assistant interactions.
    -   `config.py`: Project configuration.
    -   `etl.py`: ETL process for document embedding.
    -   `retriever.py`: Job search in a Chroma vector store.
    -   `utils.py`: Utility functions, such as PDF text extraction.
-   `dataset/`: Folder for data used in the project.
-   `tests/`: Tests for the project.
-   `requirements.txt`: Project dependencies.

## Requirements

-   Python 3.8 or higher
-   [OpenAI API Key](https://beta.openai.com/signup/)
-   Packages listed in `requirements.txt`

## Installation

1.  **Clone the repository**:
       
    `git clone https://github.com/your-username/your-project.git  
    cd your-project` 
    
2.  **Create and activate a virtual environment**:
        
    ``python -m venv venv source venv/bin/activate  # On Windows use `venv\Scripts\activate` `` 
    
3.  **Install the dependencies**:
     
    `pip install -r requirements.txt` 
    
4.  **Configure the `.env` file**: Create a `.env` file in the root of the project and add the following variables:
     
    `OPENAI_API_KEY="your_openai_api_key"
    OPENAI_LLM_MODEL="gpt-3.5-turbo-0125"` 
    
## Usage

1.  **Run the ETL Processor**:
       
    `python backend/etl.py` 
    
2.  **Start the Application**:
       
    `python backend/app.py` 

## Contact

Developed by [Gustavo Navarro](https://github.com/gnavarrolema). You can contact me at gustavo.navarrolema@gmail.com.
