import numpy as np
import math
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, optimization
from .patch import patch_hf
from .cacheManager import cacheManager
from transformers import Trainer
from torch.nn import CrossEntropyLoss
import torch


class modelManager():
    def __init__(self, args, tokenizer=None):
        #print("init model")
        self.path = args.merge.path
        if tokenizer==None:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(args.model.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
        self.model = patch_hf(self.model, args.model.type, **args.model)
        self.cachemanager = cacheManager(args=args)

    def clear(self):
        self.cachemanager.past_kv = None
        self.cachemanager.count = 0
        self.cachemanager.sum_token = 0
        

class modelInference(modelManager):
    def __init__(self, args):
        super(modelInference, self).__init__(args)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.cachemanager.model.parameters():
            param.requires_grad_(False)

    def forward(self, text=None, input_ids=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']
        with torch.inference_mode():
            result = self._decode(input_ids, **kwargs)
        return result

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])
        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()
        return model_inputs

    def generate(self, text=None, input_ids=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']
        with torch.inference_mode():
            result = self._decode(input_ids, **kwargs)
        print("result",result)    
        return result

    def _decode(self, input_ids, max_length=100, extra_end_token_ids=[], chunk_size: int = 4096, output=False):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1
        length = input_ids.size(1)
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        if output:
            output_text = ""
        for i in range(max_length + 1):
            if i == 0:
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    ed = min(input_ids.size(1) - 1, st + chunk_size)
                    if st != 0:
                        self.cachemanager.getInputs(ed - st)  
                        pass
                                         
                    out = self.model(
                        input_ids = input_ids[:, st: ed],
                        attention_mask = attention_mask[:, :ed],
                        use_cache = True,
                        return_dict = True,
                        past_key_values = self.cachemanager.past_kv
                    )
                    logits, self.cachemanager.past_kv = out.logits, out.past_key_values
                self.cachemanager.inference()
                out = self.model(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    use_cache = True,
                    return_dict = True,
                    past_key_values = self.cachemanager.past_kv
                )
                logits, self.cachemanager.past_kv = out.logits, out.past_key_values
            else:
                self.cachemanager.inference()
                out = self.model(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = self.cachemanager.past_kv,
                    use_cache = True,
                    return_dict = True
                )
                logits, self.cachemanager.past_kv = out.logits, out.past_key_values
            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_length:
                break
            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )
            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp


        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return [self.tokenizer.decode(input_ids.squeeze(0)[length:])]




class modelPretrain(modelManager, Trainer):#
    def __init__(self, args, training_args, train_dataset, 
                 eval_dataset=None, data_collator=None, tokenizer=None):
        modelManager.__init__(self, args, tokenizer)
        
        self.freeze_llm = args.freeze_llm
        self.freeze_mlp = args.freeze_mlp
        if args.freeze_llm == True and args.freeze_mlp == False:
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.cachemanager.model.parameters():
                param.requires_grad_(True)
            optimizer_grouped_parameters = [
                {"params": self.cachemanager.model.parameters()}
            ]
        elif args.freeze_llm == False and args.freeze_mlp == True:
            for param in self.model.parameters():
                param.requires_grad_(True)
            for param in self.cachemanager.model.parameters():
                param.requires_grad_(False)
            optimizer_grouped_parameters = [
                {"params": self.model.parameters()}
            ]
        elif args.freeze_llm == False and args.freeze_mlp == False:
            for param in self.model.parameters():
                param.requires_grad_(True)
            for param in self.cachemanager.model.parameters():
                param.requires_grad_(True)
            optimizer_grouped_parameters = [
                {"params": self.cachemanager.model.parameters()},
                {"params": self.model.parameters()}
            ]
        optimizer=torch.optim.AdamW(optimizer_grouped_parameters,
                                    lr=args.learning_rate)
        
        scheduler = optimization.get_cosine_schedule_with_warmup(optimizer = optimizer,
                                                                 num_warmup_steps = args.num_warmup_steps,
                                                                 num_training_steps = args.num_training_steps)
        #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.T_max, eta_min=args.learning_rate)   
        Trainer.__init__(self, model = self.model, 
                         args = training_args, 
                         optimizers = (optimizer, scheduler),
                         train_dataset = train_dataset, 
                         eval_dataset = eval_dataset, 
                         data_collator = data_collator,
                         tokenizer = self.tokenizer)

    def save_model(self, output_dir=None, _internal_call=False):
        if self.freeze_llm == False:
            self.model.save_pretrained(output_dir)
        if self.freeze_mlp == False:
            torch.save(self.cachemanager.model, output_dir+"_merge")


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.cachemanager.train(model = self.model, inputs = inputs, 
                                        train_func = self.compute_loss,
                                        optimizer = self.optimizer,
                                        scheduler = self.lr_scheduler)
        self.clear()
        return loss.detach()
    
    def validation_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        loss = self.cachemanager.train(model = self.model, inputs = inputs, 
                                 train_func = self.compute_loss)
        
        self.clear()
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        #forward
        model_output = model(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            use_cache = True,
            return_dict = True,
            past_key_values = self.cachemanager.past_kv
        )
        lm_logits, self.cachemanager.past_kv = model_output.logits, model_output.past_key_values    
        #lm_logits = nn.Softmax(dim=-1)(lm_logits)
        #compute_loss
        loss = None
        labels = inputs['input_ids']
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # shift_logits shape: [batch, sentence_length, vocab_size]，去除结束字符
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels shape: [batch, sentence_length]，去除开始字符
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss.requires_grad_(True)
            #print("true loss:",loss)
        return (loss, lm_logits) if return_outputs else loss
    

