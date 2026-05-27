import base64
from pathlib import Path
from typing import Optional
from back.evrag.config import EVRAG_CONFIG

class FrameDescriptor:
    """
    Generate textual descriptions of video frames using Vision LLMs.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or EVRAG_CONFIG
        self.llm_client = self._init_llm()

    def _init_llm(self):
        """Initialize LLM client with vision capabilities."""
        from langchain_openai import ChatOpenAI
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        return None

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def describe_frame(self, image_path: Path) -> str:
        """
        Generate a detailed description of a frame.
        """
        if not self.llm_client:
            return "LLM vision client not configured."

        base64_image = self._encode_image(image_path)
        
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe detalladamente lo que se ve en esta imagen de una clase de arquitectura de software. Enfócate en diagramas, texto en pantalla y temas tratados. Sé conciso pero técnico. Responde en español."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        try:
            response = self.llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error describiendo frame: {e}"

    async def describe_batch(self, frame_paths: list[Path]) -> list[str]:
        """Describe multiple frames."""
        descriptions = []
        for path in frame_paths:
            desc = await self.describe_frame(path)
            descriptions.append(desc)
        return descriptions
