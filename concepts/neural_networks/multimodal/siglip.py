from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiGLIPVisionConfig:

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        # Embedding Size
        self.hidden_size  = 768
        # linear layer in feed forward network
        self.intermediate_size = 3072,
        # Attention Heads
        self.num_attention_heads = 12,
        self.num_hidden_layers = 12,
        self.num_attention_heads = 12,
        # Channels in the input image : 3 (RGB)
        self.num_channels = 3,
        self.image_size = 224,
        self.patch_size = 16,
        # Layer Normalization
        self.layer_norm_eps = 1e-6,
        # self.dropout = 0.1,
        self.attention_dropout = 0.1,
        # Denotes the number of image tokens
        # How many image embeddings we will get for each image
        # (image_size / patch_size) ** 2
        self.num_img_tokens = None

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiGLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.path_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding( self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        embeddings = self.path_embedding(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embeddings(self.position_ids).unsqueeze(0)
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config : SiGLIPVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states : [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config: SiGLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, embed_dim = hidden_states.shape
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, num_heads, head_dim]
        # Splitting into Multi Heads
        query = self.query(hidden_states).view(batch_size, num_patches, self.num_heads, self.head_dim)
        key = self.key(hidden_states).view(batch_size, num_patches, self.num_heads, self.head_dim)
        value = self.value(hidden_states).view(batch_size, num_patches, self.num_heads, self.head_dim)
        
        # [batch_size, num_patches, num_heads, head_dim]-> [batch_size, num_heads, num_patches,

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Calculate Attention Score
        # Using the formulae : Attention(Q,K,V) = Q * K^T / sqrt(d_k)
        # attention weights : [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query, key.transpose(2, 3)) * self.scale)

        
        
class SiglipVisionEncoder(nn.Module):
    def __init__(self, config:  SiGLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        # residual : [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        # Skip Connections
        hidden_states = hidden_states + residual

        # residual : [batch_size, num_patches, embed_dim]
        residual = hidden_states

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        # Skip Connections
        hidden_states = hidden_states + residual
        return hidden_states




class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiGLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
    
    def forward(self,pixel_values):
        # [batch_size, num_channels, height, width] -> [batch_size, num_patches Embedding_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_sate = self.encoder(input_embeds = hidden_states)
        last_hidden_sate = self.post_layernorm(hidden_states)
        return last_hidden_sate

class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiGLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self,pixel_values):
        # [batch_size, num_channels, height, width] -> [batch_size, num_patches Embedding_dim]
        return self.vision_model(pixel_values=pixel_values)