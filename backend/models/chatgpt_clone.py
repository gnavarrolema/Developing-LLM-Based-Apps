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
            model=llm_model,  # Usar 'model' en lugar de mezclar model/model_name
            api_key=api_key,   # Usar 'api_key' consistentemente
            temperature=temperature
        )

        # Create an LLM chain with memory
        self.memory = ConversationBufferWindowMemory(k=history_length, input_key="human_input", memory_key="history")
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
            
            # Ensure consistent return format (should return the text string)
            if isinstance(response, dict) and "text" in response:
                return response["text"]
            
            # For backwards compatibility, handle both dictionary and string responses
            # For LLMChain, "text" is the default output key. "output" might be for other chains/agents.
            return response.get("text", str(response)) # Prefer "text" for LLMChain
        except Exception as e:
            # Consider logging the error here as well
            return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"


if __name__ == "__main__":
    # Create an instance of ChatAssistant with appropriate settings
    chat_assistant = ChatAssistant(
        llm_model=settings.OPENAI_LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        history_length=2,
    )

    # Use the instance to generate a response
    output = chat_assistant.predict(
        human_input="what is the answer to life the universe and everything?"
    )

    print(output)