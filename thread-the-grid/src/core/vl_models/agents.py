# # src/core/vl_models/agents.py

# from pydantic_ai import Agent
# from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.providers.openai import OpenAIProvider


# from src.settings import get_settings

# settings = get_settings()

# # configure your local Ollama model endpoint
# ollama_model = OpenAIModel(
#     model_name=settings.OLLAMA_MODEL_NAME,
#     provider=OpenAIProvider(base_url=settings.OLLAMA_PROVIDER_URL),
# )

# # xx_agent
# xx_agent = Agent(
#     ollama_model,
#     system_prompt=xx_prompt,
#     output_type=yy,
# )
