from typing import Any, Dict, List, Union
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import logging
import gc


class Model:
    def __init__(self, weights_dtype: str, inference_type: str, model_id: str, task: str, gguf_filename: str, logger: logging.Logger) -> None:
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
        self.gguf_filename = gguf_filename
        self._get_quantization_config()
        self._load_model()

    def _get_quantization_config(self):
        if self.weights_dtype.lower() == 'bnb':
            self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                            bnb_4bit_compute_dtype=torch.float16,
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_quant_type='nf4')
            self.torch_dtype = torch.float16
        elif self.weights_dtype.lower() == 'float32':
            self.quantization_config = None
            self.torch_dtype = torch.float32
        elif self.weights_dtype.lower() == 'float16':
                self.quantization_config = None
                self.torch_dtype = torch.float16
        elif self.weights_dtype.lower() == 'bfloat16':
                self.quantization_config = None
                self.torch_dtype = torch.bfloat16
        elif self.weights_dtype.lower() in ('awq', 'gptq'):
            self.quantization_config = None
            self.torch_dtype = 'auto'
        elif self.weights_dtype.lower() == 'gguf':
            raise ValueError("GGUF not supported")
        elif self.weights_dtype.lower() == '1bit':
            # not sure if is supported - maybe need to install transformers from source
            self.quantization_config = None
            self.torch_dtype = torch.float16
        else:
            self.logger.error(f"Quantization type {self.weights_dtype} not supported")
            raise ValueError(f"Quantization type {self.weights_dtype} not supported")

    def _load_model(self) -> None:
        model_args = {'device_map': 'auto', 'output_hidden_states': True, 'torch_dtype': self.torch_dtype}
        if self.quantization_config is not None:
            model_args['quantization_config'] = self.quantization_config
        if self.weights_dtype.lower() == 'gguf':
            model_args['gguf_file'] = self.gguf_filename
        print(model_args)
        try:
            if self.inference_type == 'forward':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_args).eval()
                self.model = self.model.model
            elif self.inference_type == 'generate':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_args).eval()
            self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct' if \
                                                           self.weights_dtype.lower() == '1bit'  else self.model_id,
                                                           use_fast=True)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"Model {self.model_id} loaded successfully")
            print(self.model)
            # if self.weights_dtype in ('float16', 'bfloat16', 'float32'):
            #     self.model.to_empty(device='cuda')
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_id}: {e}")
            raise Exception(f"Error loading model {self.model_id}: {e}")

    def inference(self, text: str, word_position: int = None, generation_kwards: dict = None) -> dict:
        try:
            # inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=1024).to(
            #     self.model.device)
            inputs = self.tokenizer.batch_encode_plus([text], return_tensors='pt', padding=False, truncation=True, max_length=1024,
                        return_attention_mask=True).to(self.model.device)
            
            nth_tokens = inputs.word_to_tokens(word_position) if word_position is not None else None
            if self.inference_type == 'forward':
                with torch.no_grad():
                    outputs = self.model(**inputs)
            elif self.inference_type == 'generate':
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwards, return_dict=True)
            else:
                self.logger.error(f"Inference type {self.inference_type} not supported")
                raise ValueError(f"Inference type {self.inference_type} not supported")
            return outputs, nth_tokens
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise Exception(f"Error during inference: {e}")

    def process_output_embeddings(self, outputs, sample: dict) -> dict:
        try:
            for j, emb in enumerate(outputs['hidden_states']):
                sample[f'embeddgins_{j}_mean'] = emb.mean(dim=1).squeeze().cpu().to(torch.float32).numpy() \
                                            if self.task == 'sentence' else None
                sample[f'embedding_{j}_last'] = emb[0,-1,:].cpu().to(torch.float32).numpy()
                sample[f'tokens_{j}'] = emb.squeeze().cpu().to(torch.float32).numpy() \
                                        if self.task == 'word' else None
            # final embeddings    
            sample['final_embeddings_mean'] = outputs['last_hidden_state'].mean(dim=1).squeeze().cpu().to(torch.float32).numpy() \
                                        if self.task == 'sentence' else None
            sample['final_embeddings_last'] = outputs['last_hidden_state'][0,-1,:].cpu().to(torch.float32).numpy()
            sample['tokens_final_embeddings'] = outputs['last_hidden_state'].squeeze().cpu().to(torch.float32).numpy() \
                                                if self.task == 'word' else None
            return sample
        except Exception as e:
            self.logger.error(f"Error processing output embeddings: {e}")
            raise Exception(f"Error processing output embeddings: {e}")
    
    def free_memory(self):
        if self.weights_dtype in ('float16', 'bfloat16', 'float32'):
            self.model.to('cpu')
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Memory freed")