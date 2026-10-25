import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, PretrainedConfig


class MzModelConfig(PretrainedConfig):
    model_type = "mz_model_v1"

    def __init__(self, **kwargs):
        kwargs["tie_word_embeddings"] = True # Important for weight tying
        super().__init__(**kwargs)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        context = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(context)

# Individual Transformer Block (Attention + MLP)
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attention = Attention(config)
        
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ffn_linear1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.gelu = nn.GELU()
        self.ffn_linear2 = nn.Linear(config.hidden_size * 4, config.hidden_size)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn_linear2(self.gelu(self.ffn_linear1(self.ln2(x))))
        return x

class MzModelV1(PreTrainedModel):
    config_class = MzModelConfig
    
    # Weight tying declaration
    _tied_weights_keys = {
        "lm_head.weight": "token_embeddings.weight"
    }
    
    def __init__(self, config):
        super().__init__(config)
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position, config.hidden_size)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        
        self.final_ln = nn.LayerNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight # Weight tying
        
        self.post_init() # Initialize weights

    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()

        if T > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds maximum supported length "
                f"{self.position_embeddings.num_embeddings}"
            )
        
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_ln(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Reshape tensors for standard CrossEntropyLoss calculation
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss