"""Router for RAG."""

from fastapi import APIRouter
from instructor import OpenAISchema
from pydantic import Field

from .env import env
from .router_utils import RouteErrorHandler

router = APIRouter(route_class=RouteErrorHandler)


class Completion(OpenAISchema):
  """Generated completion of a prompt."""
  completion: str = Field(...,
                          description='The answer to the question, given the context and query.')


@router.get('/generate_completion')
def generate_completion(prompt: str) -> str:
  """Generate the completion for a prompt."""
  try:
    import openai
  except ImportError:
    raise ImportError('Could not import the "openai" python package. '
                      'Please install it with `pip install openai`.')

  openai.api_key = env('OPENAI_API_KEY')
  if not openai.api_key:
    raise ValueError('The `OPENAI_API_KEY` environment flag is not set.')
  completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo-0613',
    functions=[Completion.openai_schema],
    messages=[
      {
        'role': 'system',
        'content': 'You must call the `Completion` function with the generated completion.',
      },
      {
        'role': 'user',
        'content': prompt
      },
    ],
  )
  result = Completion.from_response(completion)
  return result.completion