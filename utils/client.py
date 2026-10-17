import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
)

# ENVIRONMENT

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env"
)


def get_required_env(
    name: str,
) -> str:
    """
    Read a required environment variable.

    Raises
    ------
    ValueError
        If the environment variable is missing or empty.
    """

    value = os.getenv(
        name
    )

    if (
        value is None
        or not value.strip()
    ):
        raise ValueError(
            f"Required environment variable "
            f"'{name}' is missing or empty."
        )

    return value.strip()


# MODEL CONFIGURATION

MODEL_CONFIGS = {

    "gpt-5.4": {
        "provider":
            "azure_openai",

        "api_key_env":
            "AZURE_OPENAI_API_KEY_GPT54",

        "endpoint_env":
            "AZURE_OPENAI_ENDPOINT_GPT54",

        "deployment_env":
            "AZURE_OPENAI_DEPLOYMENT_GPT54",

        "api_version":
            "2025-04-01-preview",

        "reasoning_effort":
            "high",

        "max_tokens":
            10000,
    },

    "grok-4.20-reasoning": {
        "provider":
            "xai",

        "api_key_env":
            "XAI_API_KEY",

        "endpoint_env":
            "XAI_ENDPOINT",

        "model_env":
            "XAI_MODEL",

        "max_tokens":
            1000,

        "temperature":
            0,

        "top_p":
            1.0,
    },
}


# CREATE CLIENT

@lru_cache(
    maxsize=None
)
def create_client(
    model: str,
):
    """
    Create and cache the configured LLM client.
    """

    if model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model '{model}'. "
            f"Available models: "
            f"{list(MODEL_CONFIGS.keys())}"
        )

    config = (
        MODEL_CONFIGS[
            model
        ]
    )

    provider = (
        config[
            "provider"
        ]
    )

    # Azure OpenAI
    if provider == "azure_openai":

        kwargs = {
            "azure_deployment":
                get_required_env(
                    config[
                        "deployment_env"
                    ]
                ),

            "api_key":
                get_required_env(
                    config[
                        "api_key_env"
                    ]
                ),

            "azure_endpoint":
                get_required_env(
                    config[
                        "endpoint_env"
                    ]
                ),

            "api_version":
                config[
                    "api_version"
                ],

            "max_tokens":
                config[
                    "max_tokens"
                ],
        }

        if config.get(
            "reasoning_effort"
        ):

            kwargs[
                "reasoning_effort"
            ] = (
                config[
                    "reasoning_effort"
                ]
            )

        return AzureChatOpenAI(
            **kwargs
        )

    # xAI
    if provider == "xai":

        endpoint_url = (
            get_required_env(
                config[
                    "endpoint_env"
                ]
            )
        )

        api_key = (
            get_required_env(
                config[
                    "api_key_env"
                ]
            )
        )

        model_name = (
            get_required_env(
                config[
                    "model_env"
                ]
            )
        )

        print(
            "\nInitialising xAI Grok"
            f"\nModel: {model_name}"
            f"\nEndpoint: {endpoint_url}\n"
        )

        return ChatOpenAI(
            base_url=(
                endpoint_url
            ),
            api_key=(
                api_key
            ),
            model=(
                model_name
            ),
            max_tokens=(
                config[
                    "max_tokens"
                ]
            ),
            temperature=(
                config[
                    "temperature"
                ]
            ),
            top_p=(
                config[
                    "top_p"
                ]
            ),
        )

    raise ValueError(
        f"Unknown provider for model "
        f"'{model}': {provider}"
    )


# RESPONSE CONVERSION

def response_to_text(
    response,
) -> str:
    """
    Convert a LangChain response object into plain text.
    """

    if not hasattr(
        response,
        "content",
    ):
        return str(
            response
        ).strip()

    content = (
        response.content
    )

    # Plain string response
    if isinstance(
        content,
        str,
    ):
        return (
            content.strip()
        )

    # Structured content response
    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                str,
            ):
                text_parts.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                item_text = (
                    item.get(
                        "text"
                    )
                )

                if item_text:
                    text_parts.append(
                        str(
                            item_text
                        )
                    )

            elif hasattr(
                item,
                "text",
            ):

                text_parts.append(
                    str(
                        item.text
                    )
                )

        if text_parts:

            return (
                "\n".join(
                    text_parts
                )
                .strip()
            )

    return str(
        content
    ).strip()