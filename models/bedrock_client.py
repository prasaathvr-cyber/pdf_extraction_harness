import json
import boto3
from botocore.config import Config
from config.settings import (
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION,
    MODEL_ID, MODEL_MAX_TOKENS, MODEL_TEMPERATURE, READ_TIMEOUT
)


class BedrockClient:
    """LLM client - calls Claude via AWS Bedrock"""

    def __init__(self):
        self.model_id = MODEL_ID
        self.client   = boto3.client(
            service_name   = 'bedrock-runtime',
            region_name    = AWS_DEFAULT_REGION,
            aws_access_key_id     = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            config         = Config(read_timeout=READ_TIMEOUT)
        )

    def invoke(self, system_prompt: str, messages: list) -> str:
        """
        Call Claude via Bedrock and return the text response.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens"       : MODEL_MAX_TOKENS,
            "temperature"      : MODEL_TEMPERATURE,
            "system"           : system_prompt,
            "messages"         : messages
        }

        response = self.client.invoke_model(
            modelId     = self.model_id,
            body        = json.dumps(body),
            contentType = "application/json",
            accept      = "application/json"
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
