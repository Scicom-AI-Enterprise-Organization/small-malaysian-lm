from transformers import Qwen3ForCausalLM
import json
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from cut_cross_entropy import linear_cross_entropy

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids, 
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            
            reduction = "sum" if num_items_in_batch is not None else "mean"
            
            loss = linear_cross_entropy(
                embeddings, 
                self.lm_head.weight, 
                labels, 
                shift=True,
                impl="cce_kahan_full_c",
                reduction=reduction,
            )
            if reduction == "sum":
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

def main():
    model = Model.from_pretrained('Qwen/Qwen3-1.7B-Base')