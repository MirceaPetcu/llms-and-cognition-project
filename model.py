import torch
import logging
import gc
from transformers import (AutoTokenizer,
                        BitsAndBytesConfig,
                        AutoModelForCausalLM)



class Model:
    def __init__(self, weights_dtype: str, model_id: str, last_embedding: bool,
                mean_embedding: bool, word_embedding: bool, logger: logging.Logger) -> None:
        """
        :param weights_dtype: quantization type/ dtype
        :param model_id: huggingface model id
        :param last_embedding: whether to store last token embedding
        :param mean_embedding: whether to store mean of all token embeddings
        :param word_embedding: whether to store word-specific embeddings
        :param logger: logger object
        """
        self.weights_dtype = weights_dtype
        self.model_id = model_id
        self.logger = logger
        self.last_embedding = last_embedding
        self.mean_embedding = mean_embedding
        self.word_embedding = word_embedding
        self._get_quantization_config()
        self._load_model()

    def _get_quantization_config(self):
        """Get the quantization config based on the weights dtype"""
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
        else:
            self.logger.error(f"Quantization type {self.weights_dtype} not supported")
            raise ValueError(f"Quantization type {self.weights_dtype} not supported")

    def _load_model(self) -> None:
        """Load the model and tokenizer from Huggingface Hub"""
        model_args = {'device_map': 'auto', 'output_hidden_states': True, 'torch_dtype': self.torch_dtype}
        if self.quantization_config is not None:
            model_args['quantization_config'] = self.quantization_config
        print(model_args)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_args).eval()
            self.model = self.model.model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,
                                                           use_fast=True)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"Model {self.model_id} loaded successfully")
            print(self.model)
            print(self.model.device)
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_id}: {e}")
            raise Exception(f"Error loading model {self.model_id}: {e}")

    def inference(self, text: str, target_word: str = None) -> tuple[dict, tuple[int, int] | None]:
        """Run inference on the model for embedding extraction"""
        try:
            # inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=1024).to(
            #     self.model.device)
            # TODO: fix this
            inputs = self.tokenizer.batch_encode_plus([text],
                                                      return_tensors='pt',
                                                      padding=False,
                                                      truncation=True,
                                                      max_length=1024,
                        return_attention_mask=True).to(self.model.device)
            nth_tokens = None
            # Build your desired mapping
            for word_id in inputs.word_ids():
                if word_id is not None:
                    start, end = inputs.word_to_tokens(word_id)
                    # Get the token IDs for the word
                    token_ids = inputs['input_ids'][0, start:end]
                    # Convert the token IDs to tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                    # Join the tokens to form the word
                    word = self.tokenizer.convert_tokens_to_string(tokens)
                    if word.strip() == target_word:
                        nth_tokens = (start, end)
            if nth_tokens is None:
                self.logger.error(f"Word {target_word} not found in the text")
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs, nth_tokens
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise Exception(f"Error during inference: {e}")

    def process_output_embeddings(self, outputs, sample: dict) -> dict:
        """Get the embeddings from the model outputs"""
        try:
            for j, emb in enumerate(outputs['hidden_states']):
                sample[f'embeddgins_{j}_mean'] = emb.mean(dim=1).squeeze().cpu().to(torch.float32).numpy() \
                                            if self.mean_embedding else None
                sample[f'embedding_{j}_last'] = emb[0,-1,:].cpu().to(torch.float32).numpy() \
                                            if self.last_embedding else None
                sample[f'tokens_{j}'] = emb.squeeze().cpu().to(torch.float32).numpy() \
                                        if self.word_embedding else None
            # final embeddings    
            sample['final_embeddings_mean'] = outputs['last_hidden_state'].mean(dim=1).squeeze().cpu().to(torch.float32).numpy() \
                                        if self.mean_embedding else None
            sample['final_embeddings_last'] = outputs['last_hidden_state'][0,-1,:].cpu().to(torch.float32).numpy() \
                                        if self.last_embedding else None
            sample['tokens_final_embeddings'] = outputs['last_hidden_state'].squeeze().cpu().to(torch.float32).numpy() \
                                                if self.word_embedding else None
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