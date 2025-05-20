import sys
from pathlib import Path
import chainlit as cl

sys.path.append(str(Path(__file__).parent.parent))

# Now, import project-specific modules
from backend.models.chatgpt_clone import ChatAssistant  
from backend.models.jobs_finder import JobsFinderAssistant  
from backend.models.jobs_finder_agent import JobsFinderAgent  
from config import settings  
from backend.utils import extract_text_from_pdf  


@cl.action_callback("Select Assistant")
async def select_assistant_action(action):
    action_value = action.value
    model_instance = None

    if action_value == "Vanilla ChatGPT":
        if not settings.OPENAI_API_KEY:
            await cl.ErrorMessage("La clave API de OpenAI no está configurada. Por favor, establece la variable de entorno OPENAI_API_KEY.").send()
            return
        
        model_instance = ChatAssistant(
            llm_model=settings.OPENAI_LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        cl.user_session.set("model", model_instance)
        await cl.Message(content="Starting chat session...").send()
        return

    # Common logic for assistants requiring resume and OpenAI API Key
    if action_value in ["Jobs finder Assistant", "Jobs Agent"]:
        if not settings.OPENAI_API_KEY:
            await cl.ErrorMessage(f"La clave API de OpenAI no está configurada para {action_value}. Por favor, establece la variable de entorno OPENAI_API_KEY.").send()
            return

        files = None
        # Wait for the user to upload a file
        while files is None:
            files = await cl.AskFileMessage(
                content="Please upload your resume as PDF to begin!",
                accept=["application/pdf"],
                max_size_mb=20,
                timeout=180,
            ).send()

        uploaded_file = files[0]

        msg = cl.Message(content=f"Processing `{uploaded_file.name}`...", disable_feedback=True)
        await msg.send()

        resume_text = ""
        try:
            with open(uploaded_file.path, "rb") as f:
                resume_text = extract_text_from_pdf(f)
            msg.content = f"Procesamiento de `{uploaded_file.name}` finalizado."
            await msg.update()
        except FileNotFoundError:
            await cl.ErrorMessage(f"Error: Archivo '{uploaded_file.name}' no encontrado.").send()
            return
        except ValueError as e:
            await cl.ErrorMessage(f"Error al procesar el PDF: {str(e)}").send()
            return

        if action_value == "Jobs finder Assistant":
            model_instance = JobsFinderAssistant(
                resume=resume_text,
                llm_model=settings.OPENAI_LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
        elif action_value == "Jobs Agent":
            model_instance = JobsFinderAgent(
                resume=resume_text,
                llm_model=settings.OPENAI_LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
        
        cl.user_session.set("model", model_instance)
        await cl.Message("Now, what kind of jobs are looking for?").send()


@cl.on_chat_start
async def on_chat_start():
    actions = [
        cl.Action(
            name="Select Assistant",
            value="Vanilla ChatGPT",
            label="Vanilla ChatGPT",
        ),
        cl.Action(
            name="Select Assistant",
            value="Jobs finder Assistant",
            label="Jobs finder Assistant",
        ),
        cl.Action(
            name="Select Assistant", value="Jobs Agent", label="Jobs Agent"
        ),
    ]
    await cl.Message(
        content="Interact with this action button:", actions=actions
    ).send()


@cl.on_message
async def main(
    message: cl.Message,
):
    model = cl.user_session.get("model")
    if not model:
        await cl.Message(content="Please select an assistant first!").send()
        return

    try:
        result = model.predict(message.content)
        await cl.Message(content=result).send()
    except Exception as e:
        # Consider logging the full error for debugging purposes
        # import logging
        # logging.error(f"Error during model prediction: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"Se produjo un error al procesar tu solicitud: {str(e)}").send()