from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings


class ChatAssistant:
    def __init__(self, llm_model, api_key, temperature=0, history_length=3):
        """
        Initialize the ChatAssistant class.

        Parameters
        ----------
        llm_model : str
            The model name.

        api_key : str
            The API key for accessing the LLM model.

        temperature : float
            The temperature parameter for generating responses.

        history_length : int, optional
            The length of the conversation history to be stored in memory. Default is 3.
        """
        # String template for the chat assistant
        template = """
        Assistant: Here is the conversation history:
        {history}
        Human: {human_input}
        Assistant:
        """
        
        # Create prompt template with input variables
        self.prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

        # Create the OpenAI chat model with consistent parameters
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            temperature=temperature
        )

        # Create an LLM chain with memory
        self.memory = ConversationBufferWindowMemory(
            k=history_length, 
            input_key="human_input", 
            memory_key="history"
        )
        
        self.model = LLMChain(
            llm=self.llm, 
            prompt=self.prompt, 
            memory=self.memory,
            verbose=settings.LANGCHAIN_VERBOSE
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
        response : str
            The response from the chat assistant.
        """
        try:
            # Invoke the chain to get a response
            response = self.model.invoke({"human_input": human_input})
            
            # Return text content consistently
            if isinstance(response, dict) and "text" in response:
                return response["text"]
            
            return response.get("text", str(response))
            
        except Exception as e:
            # Log the error for debugging
            import logging
            logging.error(f"Error processing request: {str(e)}")
            return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"