from typing import List
from agents.message_data_classes import ToolResultsMessage,Message
from autogen_core.base import AgentId, MessageContext
from autogen_core.components import (
    RoutedAgent,
    message_handler,
)
from autogen_core.components.models import (
    ChatCompletionClient,
    LLMMessage,
    FunctionExecutionResult,
    SystemMessage,
    UserMessage,    
)

from autogen_ext.models import OpenAIChatCompletionClient                                                                                                  
from autogen_core.components.tool_agent import tool_agent_caller_loop
from autogen_core.components.tools import ToolSchema
from rich.console import Console
from rich.markdown import Markdown



### Base Agent defininiton:

## Base Tool Agent: 

# Takes a Promt and return the result of the Function Execution Results as a list, if it fails tool calls it will pass back the last message input as is

class BaseToolsAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        system_message: str,
        tools: list[ToolSchema],
        tool_agent_type: str,
    ) -> None:
        super().__init__(description=description)
        self._model_client=model_client
        self._system_message=SystemMessage(system_message)
        self._tools = tools
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)

        
#by default it takes only a message input and returns an execution return if called by user message, dosnt keep track of the conversation if a purely functional agent. 
# if you want to add more capabilities, you need to implement it in the child class:
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) ->  ToolResultsMessage:
        Console().print(Markdown(f"### {self.id.type}: "))
        # Create a session of messages and a result variable:
        session: List[LLMMessage] = [UserMessage(content=message.content,source="User")] #u can add a system message to this list at the beggining but i cannot find it any usefull to do so (is basically chaining system promts.)
        # Run the caller loop to handle tool calls.
        messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,        
            tool_schema=self._tools,
            cancellation_token=ctx.cancellation_token,
        )
        # Return the final response list and filters the tool generation results to send them to a tool result processing function it messages the Agregator to saves results .
        results = []
        for call in messages:
            # Check if call.content is a list
            if isinstance(call.content, list):              
                for item in call.content:
                    if isinstance(item, FunctionExecutionResult):
                        #print(f"Processing FunctionExecutionResult: - Content: {item.content}..."[:550])
                        results.append(item.content)                                              
            elif isinstance(call.content, str) and isinstance(call, FunctionExecutionResult):
                results = [call.content]           
        
        if results:
            Console().print(f"Execution Result: {results}")
            return ToolResultsMessage(content=results, source=self.id.key) 
        # Return the final message or the message with the next result
        Console().print(f"No tools were called, returning original message: {message.content[:200]}\n\n")
        return ToolResultsMessage(content=[message.content], source=self.id.key) 


## Base Basic Agent is a simple Request - Response Agent it just do that takes a request and returns a response as a value without broadcasting further.

class BaseBasicAgent(RoutedAgent):
    def __init__(
            self,
            description: str,
            model_client: ChatCompletionClient,
            system_message: str,
        ) -> None:
            super().__init__(description=description)
            self._model_client = model_client
            self._system_message = SystemMessage(system_message)
            self._chat_history: List[LLMMessage] = []


    #by default, it recieves a message and a next recipient to be called uppon completion, ovveride this to handle more complex cases, this is basically a round robin, 
    # the limitation lies that it will only keep track of the messages recieved by him directly and his responses, if you need more complex message history management override this handler.
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.extend(
            [
                UserMessage(content=f"Last Message:{message.content}", source=f"{message.source}"), 
            ]
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        self._chat_history.append(UserMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content,"\n\n"))
        results = Message(content=completion.content, source=self.id.type)
        return results

