Este proyecto implementa una aplicación de chat interactiva construida con Chainlit, que ofrece varios asistentes impulsados por Modelos de Lenguaje Grandes (LLMs). Incluye un asistente de chat genérico tipo ChatGPT, un buscador de empleos que utiliza RAG (Retrieval Augmented Generation) sobre una base de datos de ofertas de trabajo, y un agente de búsqueda de empleos.

## Características Principales

   **Múltiples Asistentes:**
   *   **Vanilla ChatGPT:** Un clon básico de ChatGPT para conversaciones generales.
   *   **Jobs finder Assistant:** Un asistente que te ayuda a encontrar ofertas de trabajo relevantes basadas en tu currículum y tus preferencias. Utiliza una base de datos vectorial de ofertas de empleo.
   *   **Jobs Agent:** Un agente más avanzado para la búsqueda de empleos (la funcionalidad específica del agente dependerá de su implementación en `models/jobs_finder_agent.py`).
   **Carga de Currículum:** Los asistentes de búsqueda de empleo permiten cargar un currículum en formato PDF para personalizar las búsquedas.
   **Proceso ETL:** Incluye un script para procesar un dataset de ofertas de trabajo (CSV), generar embeddings y almacenarlos en una base de datos vectorial ChromaDB.
   **Interfaz de Usuario con Chainlit:** Interfaz de chat amigable y fácil de usar.
   **Configuración Flexible:** Utiliza variables de entorno y un archivo `.env` para la configuración.

## Estructura del Proyecto

```
LLM_App-Gustavo_Navarro/
 ├── backend/
 │   ├── __init__.py
 │   ├── app.py            # Aplicación principal Chainlit
 │   ├── config.py         # Configuración del proyecto
 │   ├── etl.py            # Script para el proceso ETL (cargar datos a ChromaDB)
 │   ├── models/           # Módulos de los asistentes
 │   │   ├── __init__.py
 │   │   ├── chatgpt_clone.py
 │   │   ├── jobs_finder.py
 │   │   └── jobs_finder_agent.py
 │   ├── retriever.py      # Lógica para buscar en ChromaDB
 │   └── utils.py          # Funciones de utilidad (ej. extracción de PDF)
 ├── chroma/               # Directorio para la base de datos ChromaDB (creado por ETL)
 ├── dataset/
 │   └── jobs.csv          # Dataset de ejemplo con ofertas de trabajo
 ├── .env.example          # Ejemplo de archivo de configuración de entorno
 ├── requirements.txt      # Dependencias del proyecto
 └── README.md             # Este archivo
```

## Requisitos Previos

*   Python 3.8 o superior
*   Una clave API de OpenAI (si se utiliza el modelo de OpenAI)

## Configuración

1.  **Clonar el repositorio (si aplica):**
    ```bash
    git clone https://github.com/gnavarrolema/Developing-LLM-Based-Apps
    cd LLM_App-Gustavo_Navarro
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
+
3.  **Instalar dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con todas las dependencias. Un ejemplo podría ser:
    ```txt
    # requirements.txt
    chainlit
    langchain
    langchain-openai
    langchain-community
    pandas
    pypdf2
    sentence-transformers
    chromadb
    python-dotenv
    pydantic-settings
    tqdm
    # Añade otras dependencias específicas de tus modelos si es necesario
    ```
    Luego instala con:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar variables de entorno:**
    Copia el archivo `.env.example` a `.env` y edítalo con tus configuraciones:
    ```bash
    cp .env.example .env
    ```
    Contenido del `.env` (ejemplo):
    ```env
    OPENAI_API_KEY="tu_clave_api_de_openai"
    OPENAI_LLM_MODEL="gpt-3.5-turbo"
    # Otras configuraciones de config.py pueden ser sobrescritas aquí si es necesario
    ```

5.  **Dataset:**
    Asegúrate de que el archivo `dataset/jobs.csv` exista y tenga el formato esperado por `backend/etl.py` (columnas: `description`, `Employment type`, `Seniority level`, `company`, `location`, `post_url`, `title`).

## Uso

### 1. Ejecutar el Proceso ETL

Este paso es necesario para crear la base de datos vectorial con las ofertas de trabajo. Ejecuta el siguiente comando desde el directorio raíz del proyecto (`LLM_App-Gustavo_Navarro/`):

```bash
python -m backend.etl --batch-size 32 --chunk-size 500 --chunk-overlap 100
```

*   `--batch-size`: Número de documentos a procesar en cada lote.
*   `--chunk-size`: Tamaño de los trozos en los que se dividen los documentos.
*   `--chunk-overlap`: Superposición entre trozos.
*   `--limit` (opcional): Limita el número de documentos a procesar del CSV (ej. `--limit 1000`).

Esto creará (o actualizará) la base de datos en el directorio `chroma/`.

### 2. Ejecutar la Aplicación Chainlit

Una vez que el proceso ETL haya finalizado (o si solo quieres usar el "Vanilla ChatGPT"), puedes iniciar la aplicación Chainlit. Ejecuta el siguiente comando desde el directorio raíz del proyecto:

```bash
chainlit run backend/app.py -w
```

*   `backend/app.py`: Es el punto de entrada de tu aplicación Chainlit.
*   `-w` (o `--watch`): Habilita la recarga automática cuando se detectan cambios en los archivos.

Abre tu navegador y ve a la dirección que Chainlit indique (normalmente `http://localhost:8000`).

## Tecnologías Utilizadas

*   **Python**
*   **Chainlit:** Para la interfaz de usuario de chat.
*   **Langchain:** Framework para construir aplicaciones con LLMs.
*   **OpenAI API:** (Opcional, para los modelos de OpenAI).
*   **Sentence Transformers:** Para la generación de embeddings.
*   **ChromaDB:** Base de datos vectorial para almacenar y buscar embeddings.
*   **Pandas:** Para la manipulación de datos (en el ETL).
*   **PyPDF2:** Para la extracción de texto de archivos PDF.
*   **Pydantic:** Para la validación de configuraciones.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cambios.

---
*Creado por Gustavo Navarro*
```
