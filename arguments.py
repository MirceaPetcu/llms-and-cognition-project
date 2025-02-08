from dataclasses import dataclass
from typing import List

@dataclass
class Arguments:
    hf_token: str
    model: str
    dtype: str
    data: str
    inference_type: str
    additional_prompt: str
    gguf_filename: str
    data_keyword: str
    text_column: str
    target_column: List[str]
    lang_column: str
    id_column: str
    word_column: str
    task: str

