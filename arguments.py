from dataclasses import dataclass
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError, validator, model_validator
import os

class PreprocessingArguments(BaseModel):
    """
    Arguments for the model processing.
    """

    hf_token: str = Field(
        description="Huggingface token for authentication",
        default=""
    )
    model: str = Field(
        description="Model identifier on Hugging Face",
        default=""
    )
    dtype: Literal["float32", "float16", "gptq", "bnb", "awq"] = Field(
        description="Quantization type for model loading",
        default=""
    )
    data: str = Field(
        description="Relative dataset path",
        default=""
    )
    additional_prompt: str = Field(
        description="Additional prompt text to prepend to inputs",
        default=""
    )
    data_keyword: str = Field(
        description="Dataset name/identifier for output naming",
        default=""
    )
    text_column: str = Field(
        description="Name of the column containing the text in the dataset",
        default=""
    )
    target_column: List[str] = Field(
        description="Names of target columns in the dataset",
        default_factory=list
    )
    lang_column: str = Field(
        description="Name of the language column in the dataset",
        default=""
    )
    id_column: str = Field(
        description="Name of the ID column in the dataset",
        default=""
    )
    word_column: str = Field(
        description="Name of the column containing target word",
        default=""
    )
    last_embedding: bool = Field(
        description="Whether to store the last token embedding",
        default=True
    )
    mean_embedding: bool = Field(
        description="Whether to store mean of all token embeddings",
        default=True
    )
    word_embedding: bool = Field(
        description="Whether to store word-specific embeddings",
        default=True
    )
    dataset_config_file: str = Field(
        description="Path to dataset configuration JSON file",
        default=''
    )

    @model_validator(mode='before')
    @classmethod
    def validate_word_embedding(cls, values):
        word_embedding = values.get('word_embedding', False)
        word_column = values.get('word_column', None)
        if word_embedding and not word_column:
            raise ValueError("word_column must be specified if word_embedding is True")
        return values

    @validator('data_keyword')
    def validate_data_keyword(cls, v):
        if not v:
            raise ValueError("data_keyword cannot be empty")
        return v
        
    @validator('target_column')
    def validate_target_column(cls, v):
        if not v:
            raise ValueError("target_column cannot be empty")
        return v

    @validator('model')
    def validate_model(cls, v):
        if not v:
            raise ValueError("model identifier cannot be empty")
        return v

    @validator('data')
    def validate_data_path(cls, v):
        if not v:
            raise ValueError("data path cannot be empty")
        return v
    
    @validator('hf_token')
    def validate_hf_token(cls, v):
        if not v:
            raise ValueError("Huggingface token cannot be empty")
        return v

    @validator('dtype')
    def validate_dtype(cls, v):
        valid_dtypes = ["float32", "float16", "gptq", "bnb", "awq"]
        if v not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")
        return v

    @validator('text_column')
    def validate_text_column(cls, v):
        if not v:
            raise ValueError("text_column cannot be empty")
        return v

    @validator('dataset_config_file')
    def validate_config_file(cls, v):
        if v and not v.endswith('.json'):
            raise ValueError("Dataset config file must have .json extension")
        return v


