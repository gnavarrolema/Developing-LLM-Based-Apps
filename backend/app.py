import sys
from pathlib import Path

import chainlit as cl

from models.chatgpt_clone import ChatAssistant
from models.jobs_finder import JobsFinderAssistant
from models.jobs_finder_agent import JobsFinderAgent

# Needed for the import of config
sys.path.append(str(Path(__file__).parent.parent))

from config import settings  # noqa: E402
from utils import extract_text_from_pdf  # noqa: E402


@cl.action_callback("Select Assistant")
async def select_assistant_action(action):
    if action.value == "Vanilla ChatGPT":
        model = ChatAssistant(
            llm_model=settings.OPENAI_LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        cl.user_session.set("model", model)
        await cl.Message(content="Starting chat session...").send()
    else:
        files = None
        # Wait for the user to upload a file
        while files is None:
            files = await cl.AskFileMessage(
                content="Please upload your resume as PDF to begin!",
                accept=["application/pdf"],
                max_size_mb=20,
                timeout=180,
            ).send()

        file = files[0]

        msg = cl.Message(
            content=f"Processing `{file.name}`...", disable_feedback=True
        )
        await msg.send()

        resume = extract_text_from_pdf(open(file.path, "rb"))
        await cl.Message(
            content=f"Finished parsing your resume, file content: {resume}"
        ).send()

        if action.value == "Jobs finder Assistant":
            model = JobsFinderAssistant(
                resume=resume,
                llm_model=settings.OPENAI_LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
        else:
            model = JobsFinderAgent(
                resume=resume,
                llm_model=settings.OPENAI_LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )

        cl.user_session.set("model", model)
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

    result = model.predict(message.content)
    await cl.Message(content=result).send()