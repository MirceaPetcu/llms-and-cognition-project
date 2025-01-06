from typing import Any, Dict, List, Union
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AwqConfig
import logging
import gc


class Model:
    def __init__(self, weights_dtype: str, inference_type: str, model_id: str, task: str, logger: logging.Logger) -> None:
        """
        :param weights_dtype: quantization type/ dtype
        :param inference_type: forward or generate
        :param model_id: huggingface model id
        :param task: sentence or word complexity prediction
        """
        self.weights_dtype = weights_dtype
        self.inference_type = inference_type
        self.model_id = model_id
        self.task = task
        self.logger = logger
        self._get_quantization_config()
        self._load_model()

    def _get_quantization_config(self):
        if self.weights_dtype.lower() == 'bnb':
            self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                            bnb_4bit_compute_dtype=torch.float16,
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_quant_type='nf4')
            self.torch_dtype = torch.float16
        elif self.weights_dtype.lower() == 'float16':
                self.quantization_config = None
                self.torch_dtype = torch.float16
        elif self.weights_dtype.lower() == 'bfloat16':
                self.quantization_config = None
                self.torch_dtype = torch.bfloat16
            # TODO(cezieu/maria19ioana): Add support for other quantization types (e.g. gptq, awq, llama.cpp etc.)
        elif self.weights_dtype.lower() == 'awq': #added awq support
                self.quantization_config = AwqConfig(
                    bits=4,
                    enable_quantization=True,
                    pre_quantized=True,
                    compute_dtype=torch.float16,
                    quantization_method="linear",
                    tokenizer=AutoTokenizer.from_pretrained(self.model_id),
                    optimize_model=True
                )
                self.torch_dtype = torch.float16
        else:
                self.logger.error(f"Quantization type {self.weights_dtype} not supported")
                raise ValueError(f"Quantization type {self.weights_dtype} not supported")

    def _load_model(self):
        try:
            if self.inference_type == 'forward':
                self.model = AutoModel.from_pretrained(self.model_id,
                                                       low_cpu_mem_usage=True,
                                                       device_map='auto',
                                                       output_hidden_states=True,
                                                       quantization_config=self.quantization_config,
                                                       torch_dtype=self.torch_dtype
                                                       ).eval()
            elif self.inference_type == 'generate':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                                  low_cpu_mem_usage=True,
                                                                  device_map='auto',
                                                                  output_hidden_states=True,
                                                                  quantization_config=self.quantization_config,
                                                                  torch_dtype=self.torch_dtype
                                                                  ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"Model {self.model_id} loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_id}: {e}")
            raise

    def inference(self, text: str, generation_kwards: dict = None) -> dict:
        try:
            inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=1024).to(
                self.model.device)
            if self.inference_type == 'forward':
                with torch.no_grad():
                    outputs = self.model(**inputs)
            elif self.inference_type == 'generate':
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwards, return_dict=True)
            else:
                self.logger.error(f"Inference type {self.inference_type} not supported")
                raise ValueError(f"Inference type {self.inference_type} not supported")
            return outputs
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise

    def process_output_embeddings(self, outputs, sample: dict) -> dict:
        try:
            for j, emb in enumerate(outputs['hidden_states']):
                sample[f'embeddgins_{j}_mean'] = emb.mean(dim=1).squeeze().cpu().to(torch.float32).numpy() \
                                            if self.task == 'sentence' else None
                sample[f'embedding_{j}_last'] = emb[-1].squeeze().cpu().to(torch.float32).numpy()
                sample[f'tokens_{j}'] = emb.squeeze().cpu().to(torch.float32).numpy() \
                                        if self.task == 'word' else None
            # final embeddings    
            sample['final_embeddings_mean'] = outputs['last_hidden_state'].mean(dim=1).squeeze().cpu().to(torch.float32).numpy() \
                                        if self.task == 'sentence' else None
            sample['final_embeddings_last'] = outputs['last_hidden_state'][-1].squeeze().cpu().to(torch.float32).numpy()
            sample['tokens_final_embeddings'] = outputs['last_hidden_state'].squeeze().cpu().to(torch.float32).numpy() \
                                                if self.task == 'word' else None
            return sample
        except Exception as e:
            self.logger.error(f"Error processing output embeddings: {e}")
            raise ValueError(f"Error processing output embeddings: {e}")
    
    def free_memory(self):
        if self.weights_dtype in ('float16', 'bfloat16', 'float32'):
            self.model.to('cpu')
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Memory freed")