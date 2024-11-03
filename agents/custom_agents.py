
from autogen_core.components.tools import ToolSchema
from autogen_core.components.models import (
    ChatCompletionClient,
)

from agents.base_agents import (
    BaseBasicAgent,
    BaseToolsAgent
)


class PromtEnhancerAgent(BaseBasicAgent):
    def __init__(self, description: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            system_message="""You are a prompt enhancer designed to take an initial user prompt and expand it into a clearer, LLM-friendly version that retains the original intent and focus. Your goal is to produce a refined paraphrase that:

1. **Enhances Clarity**: Rephrase complex or vague language into precise, easy-to-understand expressions.
2. **Improves Readability**: Ensure the prompt is formatted for smooth reading and natural flow without introducing unnecessary complexity or length.
3. **Preserves Intent**: Maintain the original purpose, tone, and goals specified by the user, while enhancing logical structure and conciseness.
4. **Mantain Format**: the promt generated will be used as is, refrain to output comments like "This paraphrased prompt: ... " or any other meta commentary 
5. ""If promted with a vague story, expand it
6- DO NOT output any meta comentaty or refer to your task
7- provide guidelines if the task is extense, such if a its generating a novel, provide a crude outline
8- DO NOT explain the changes to the previus prompts.
When rephrasing, avoid making assumptions, changing meanings, or adding details not implied in the original text. Keep your response well-balanced: informative yet concise.""",)
        



class WebSearchAgent(BaseToolsAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        tools: ToolSchema,
        tool_agent_type: str,
    ) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            tools=tools,
            tool_agent_type = tool_agent_type,
            system_message="""Input: Accepts a chapter or descriptive passage from a book. This could include setting details, character descriptions, 
and any relevant objects or themes to include in the visual representation.

Objective: To use the given passage and generate an FLUX model-based image that captures the essence and atmosphere of the text provided. 
The image should emphasize the scene's visual mood, key elements, and any unique qualities described in the passage.

Instructions:

Analyze the passage for details about the setting, characters, and any thematic elements.
Identify key descriptors (e.g., colors, lighting, character emotions, objects, and backgrounds).
Frame the scene, paying close attention to the chapter's specific mood (e.g., tense, serene, eerie) and narrative style (e.g., high fantasy, urban, sci-fi).
Execution: Use the generate_image tool with the FLUX model, ensuring the prompt is detailed and focuses on creating a visually compelling 
scene that matches the chapter's imagery. For instance, describe lighting, key actions, and any relevant background elements 
to bring the scene to life. The resulting image should be detailed, high-quality, and true to the original text.

Example:

If the passage describes a dense forest where a character discovers a mysterious, glowing artifact, the agent would prompt the SDXL model with: 
"An eerie, dark forest at twilight with thick, twisted trees and fog drifting between them. In the center, a small, 
radiant artifact glowing in pale blue light lies on the forest floor, casting an ethereal glow. The scene feels mysterious 
and slightly ominous, with dark shadows and subtle details of twisted roots and moss-covered ground."""
        )
