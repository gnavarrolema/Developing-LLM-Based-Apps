from unittest.mock import MagicMock, patch

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

from backend.config import settings
from backend.models.jobs_finder import JobsFinderAssistant
from backend.models.jobs_finder_agent import JobsFinderAgent


@patch("backend.models.jobs_finder.resume_summarizer")
def test_jobs_finder_agent(resume_summarizer_chain_mock):
    resume_summarizer_chain_mock.invoke.return_value = "Resume summary"

    job_finder_agent = JobsFinderAgent(
        resume="resume",
        llm_model="gpt-3.5-turbo",
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        history_length=2,
    )

    # JobsFinderAgent attributes
    assert hasattr(job_finder_agent, "resume")
    assert hasattr(job_finder_agent, "llm")
    assert hasattr(job_finder_agent, "job_finder")
    assert hasattr(job_finder_agent, "agent_executor")
    assert hasattr(job_finder_agent, "memory")  # Corregido: Ahora usa memory, no agent_memory
    assert hasattr(job_finder_agent, "history_length")
    assert callable(job_finder_agent.predict)

    # JobsFinderAgent attribute types
    assert isinstance(job_finder_agent.llm, ChatOpenAI)
    assert isinstance(job_finder_agent.job_finder, JobsFinderAssistant)
    assert isinstance(job_finder_agent.agent_executor, AgentExecutor)
    assert isinstance(job_finder_agent.memory, ConversationBufferWindowMemory)  # Corregido
    assert isinstance(job_finder_agent.history_length, int)

    # Verificar que el resume_summarizer fue llamado correctamente
    # Actualizado para usar el comportamiento mock correcto
    assert job_finder_agent.job_finder.resume_summary == "Resume summary"

    # AgentExecutor attributes tools
    assert len(job_finder_agent.agent_executor.tools) == 2
    assert job_finder_agent.agent_executor.tools[0].name == "jobs_finder"
    assert job_finder_agent.agent_executor.tools[1].name == "cover_letter_writing"
    
    # Verificar que la memoria tiene la configuración correcta
    assert job_finder_agent.memory.k == 2
    assert job_finder_agent.memory.memory_key == "chat_memory"
    assert job_finder_agent.memory.output_key == "output"
    assert job_finder_agent.memory.return_messages == True