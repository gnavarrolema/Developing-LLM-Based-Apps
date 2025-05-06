from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from backend.config import settings
from backend.models.jobs_finder import JobsFinderAssistant


def build_job_finder(job_finder_assistant):
    def job_finder(human_input: str):
        return job_finder_assistant.predict(human_input)
    return job_finder


def build_cover_letter_writing(llm, resume):
    def cover_letter_writing(job_description: str):
        template = """
        You are an AI assistant tasked with writing a cover letter for a job application.
        
        Resume:
        {resume}
        
        Job Description:
        {job_description}
        
        Using the skills and experience mentioned in the resume, write a compelling cover letter tailored to the job description.
        """
        
        prompt = PromptTemplate(
            input_variables=["resume", "job_description"],
            template=template,
        )
        
        cover_letter_writing_chain = LLMChain(llm=llm, prompt=prompt)
        return cover_letter_writing_chain.invoke(resume=resume, job_description=job_description)
    return cover_letter_writing


class JobsFinderAgent:
    def __init__(self, resume, llm_model, api_key, temperature=0, history_length=3):
        """
        Initialize the JobsFinderAgent class.
        
        Parameters
        ----------
        resume : str
            The resume of the user.
            
        llm_model : str
            The model name.
            
        api_key : str
            The API key for accessing the LLM model.
            
        temperature : float
            The temperature parameter for generating responses.
            
        history_length : int, optional
            The length of the conversation history to be stored in memory. Default is 3.
        """
        self.llm_model = llm_model
        self.resume = resume
        self.llm = ChatOpenAI(model=llm_model, api_key=api_key, temperature=temperature)

        # Create the Job finder tool
        self.job_finder = JobsFinderAssistant(
            resume=resume,
            llm_model=llm_model,
            api_key=api_key,
            temperature=temperature,
        )

        # Usar memoria integrada de LangChain
        self.memory = ConversationBufferWindowMemory(
            k=history_length,
            return_messages=True,
            memory_key="chat_memory",
            output_key="output"
        )
        
        self.agent_executor = self.create_agent()
        self.history_length = history_length

    def create_agent(self):
        job_finder = build_job_finder(self.job_finder)
        cover_letter_writing = build_cover_letter_writing(self.llm, self.resume)

        tools = [
            Tool(
                name="jobs_finder",
                func=job_finder,
                description="Look up for jobs based on user preferences.",
                handle_tool_error=True,
            ),
            Tool(
                name="cover_letter_writing",
                func=cover_letter_writing,
                description="Write a cover letter based on a job description, extract as much information as you can about the job from the user input and from the chat history.",
                handle_tool_error=True,
            ),
        ]

        try:
            prompt = hub.pull("hwchase17/openai-functions-agent")
        except Exception as e:
            # Usar un prompt predeterminado si falla la llamada al hub
            prompt = PromptTemplate.from_template(
                "You are a helpful AI assistant that helps people find jobs.\n"
                "Human: {input}\n"
                "AI: "
            )
            # Registrar el error para fines de depuración
            import logging
            logging.error(f"Error pulling prompt from hub: {str(e)}")

        agent = create_openai_functions_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=settings.LANGCHAIN_VERBOSE,
            early_stopping_method="force",
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            memory=self.memory  
        )

    def predict(self, human_input: str) -> str:
        """
        Generate a response to a human input.
        
        Parameters
        ----------
        human_input : str
            The human input to the chat assistant.
            
        Returns
        -------
        str
            The response from the chat assistant.
        """
        try:
            # Usar memoria integrada de LangChain
            agent_response = self.agent_executor.invoke(
                {"input": human_input}
            )
            
            return agent_response["output"]
            
        except Exception as e:
            import logging
            logging.error(f"Error in agent execution: {str(e)}")
            return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"