from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings

# Crear una plantilla de cadena para esta cadena. Debe indicar al LLM
# que se proporciona un currículum para ser resumido para extraer las habilidades del candidato.
# La plantilla debe tener una variable de entrada: `resume`.
template = """Below a resume is provided. Please summarize the resume extracting the main skills and the years of experience in the field. Also, extract the country of residence and the speaking languages.

Resume:
{resume}

Summary:
"""

def get_resume_summarizer_chain():
    # Crear una plantilla de prompt usando la cadena de plantilla creada anteriormente.
    prompt_template = PromptTemplate(
        input_variables=["resume"],
        template=template,
    )

    # Crear una instancia de `langchain.chat_models.ChatOpenAI` con la configuración adecuada.
    llm = ChatOpenAI(
        model = settings.OPENAI_LLM_MODEL,
        api_key = settings.OPENAI_API_KEY,
        temperature=0
    )

    # Crear una instancia de `langchain.chains.LLMChain` con la configuración adecuada.
    resume_summarizer_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=settings.LANGCHAIN_VERBOSE
    )

    return resume_summarizer_chain

if __name__ == "__main__":
    resume_summarizer_chain = get_resume_summarizer_chain()
    print(
        resume_summarizer_chain.invoke(
            {"resume": "I am a software engineer with 5 years of experience"}
        )
    )