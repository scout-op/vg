# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Lightweight Transformer Decoder for SegVG.

This implementation includes several optimizations:
1. Reduced number of layers (from 6 to 3)
2. Multi-stage interaction design
3. Feature chunking for attention computation
4. Parameter sharing across layers
5. Low-rank approximation for attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from ..vl_transformer import MultiheadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Any # Added Any for _get_activation_fn flexibility

class VisionLanguageDecoder(nn.Module):
    """
    Decoder module that stacks multiple VltvgStyleDecoderLayers.
    Accepts separate text and visual memory inputs.
    """
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 img_feat_chunk_num=1): # Added img_feat_chunk_num
        super().__init__()

        # Instantiate the new VLTVG-style layer
        decoder_layer = VltvgStyleDecoderLayer(d_model, nhead, dim_feedforward,
                                               dropout, activation, img_feat_chunk_num)

        # Use the existing TransformerDecoder wrapper (which we will also modify)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                # --- Query Inputs ---
                tgt: Tensor,                          # The input query sequence (e.g., object queries)
                query_pos: Optional[Tensor] = None,   # Positional encoding for the query

                # --- Text Memory Inputs ---
                text_memory: Optional[Tensor] = None, # Encoded text features
                text_memory_key_padding_mask: Optional[Tensor] = None, # Padding mask for text
                text_pos: Optional[Tensor] = None,    # Positional encoding for text

                # --- Visual Memory Inputs ---
                visual_memory: Optional[Tensor] = None, # Encoded visual features
                visual_memory_key_padding_mask: Optional[Tensor] = None, # Padding mask for visual
                visual_pos: Optional[Tensor] = None,    # Positional encoding for visual

                # --- Control ---
                layer_idx: Optional[int] = None       # Optional: compute only a specific layer
               ):
        """
        Forward pass for the VL Decoder.
        Args now distinguish between text and visual memory components.
        """
        # Pass all the distinct arguments to the underlying TransformerDecoder
        return self.decoder(
            tgt=tgt,
            query_pos=query_pos,
            text_memory=text_memory,
            text_memory_key_padding_mask=text_memory_key_padding_mask,
            text_pos=text_pos,
            visual_memory=visual_memory,
            visual_memory_key_padding_mask=visual_memory_key_padding_mask,
            visual_pos=visual_pos,
            layer_idx=layer_idx
        )

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Assume _get_activation_fn is defined elsewhere, like this placeholder:
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# Assume MultiheadAttention is defined elsewhere, likely nn.MultiheadAttention
# If VLTVG used a custom one, you might need to import/define that.
# Using nn.MultiheadAttention for now based on your original code.

class VltvgStyleDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer modified to follow the VLTVG MultiStage approach.
    1. Attends to Text Memory using the input query.
    2. Attends to Visual Memory using the text-informed query.
    3. Passes through FFN.
    Uses Post-Normalization similar to VLTVG's structure.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", img_feat_chunk_num=1):
        super().__init__()
        # --- Attention Modules ---
        # Attention for Text Memory (Word Attention in VLTVG)
        self.text_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False) # Assuming batch_first=True is desired
        # Attention for Visual Memory (Image Attention in VLTVG)
        self.visual_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False) # Assuming batch_first=True is desired

        # --- Feedforward Network ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) # Dropout within FFN
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

        # --- Normalization and Dropout Layers ---
        # VLTVG uses 3 norms and 3 dropouts, applied post-operation typically
        self.norm1 = nn.LayerNorm(d_model) # After text attention
        self.norm2 = nn.LayerNorm(d_model) # After visual attention + residual
        self.norm3 = nn.LayerNorm(d_model) # After FFN + residual

        self.dropout1 = nn.Dropout(dropout) # After text attention output before norm
        self.dropout2 = nn.Dropout(dropout) # After visual attention output before residual add
        self.dropout3 = nn.Dropout(dropout) # After FFN output before residual add

        # --- Configuration ---
        self.img_feat_chunk_num = img_feat_chunk_num # For potential visual feature splitting

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Adds positional embedding to the tensor."""
        return tensor if pos is None else tensor + pos

    def forward(self,
                # --- Main Input Query ---
                query: Tensor,                       # Analogous to 'tgt' in original or 'vis_query' in VLTVG
                query_pos: Optional[Tensor] = None,  # Positional encoding for the input query

                # --- Text Memory Inputs ---
                text_memory: Optional[Tensor] = None, # Text features (e.g., from text encoder)
                text_memory_key_padding_mask: Optional[Tensor] = None, # Padding mask for text
                text_pos: Optional[Tensor] = None,   # Positional encoding for text memory

                # --- Visual Memory Inputs ---
                visual_memory: Optional[Tensor] = None, # Visual features (e.g., from image backbone/encoder)
                visual_memory_key_padding_mask: Optional[Tensor] = None, # Padding mask for visual features
                visual_pos: Optional[Tensor] = None,   # Positional encoding for visual memory

                # --- Optional Masks (if needed, though text/visual masks cover most cases) ---
                # tgt_mask: Optional[Tensor] = None, # Self-attention mask (Not used in VLTVG style)
                # memory_mask: Optional[Tensor] = None, # General memory mask (split into text/visual)

                # --- Optional Positional Encoding (VLTVG uses separate pos for visual query) ---
                # If the query pos embedding should change after text attention:
                text_informed_query_pos: Optional[Tensor] = None
               ):
        """
        Forward pass implementing the VLTVG-style multi-stage attention.
        Args are adjusted for clarity between text and visual inputs.
        Note: Assumes batch_first=True for MultiheadAttention layers based on common practice.
              If your original code or data is (Seq, Batch, Dim), set batch_first=False.
        """
        # --- Feature Chunking (Optional, from VLTVG) ---
        if visual_memory is not None and self.img_feat_chunk_num > 1:
            # Assuming visual_memory has shape (Batch, SeqLen, Dim * chunk_num) or similar
            # Adjust dim splitting based on your actual implementation
            visual_memory_srcs = visual_memory.chunk(self.img_feat_chunk_num, dim=-1)
            visual_memory_k = visual_memory_srcs[1] # Example: Key uses second chunk
            visual_memory_v = visual_memory_srcs[0] # Example: Value uses first chunk
            # If using pos encodings per chunk, they might need splitting too
        else:
            visual_memory_k = visual_memory_v = visual_memory

        # 1. --- Text Attention ---
        # Query attends to Text Memory to gather linguistic context.
        q_text = self.with_pos_embed(query, query_pos)
        k_text = self.with_pos_embed(text_memory, text_pos)
        v_text = text_memory

        # Pass through text attention
        # Note: nn.MultiheadAttention returns (attn_output, attn_output_weights)
        # We only need the output here.
        text_info, _ = self.text_attn(query=q_text, key=k_text, value=v_text,
                                      key_padding_mask=text_memory_key_padding_mask)
                                      # attn_mask=None - usually no mask for cross-attn unless causal

        # Apply dropout and normalization (Post-Attention Norm)
        # VLTVG applies norm after dropout: intermediate_query = norm(dropout(text_info))
        intermediate_query = self.norm1(self.dropout1(text_info))

        # 2. --- Visual Attention ---
        # The text-informed query now attends to Visual Memory.
        # Determine the query positional encoding for this stage.
        # Use text_informed_query_pos if provided, otherwise reuse query_pos or intermediate_query's implicit pos.
        # VLTVG uses a potentially different pos ('text_query_pos' in their code).
        q_visual_pos = text_informed_query_pos if text_informed_query_pos is not None else query_pos
        q_visual = self.with_pos_embed(intermediate_query, q_visual_pos)

        k_visual = self.with_pos_embed(visual_memory_k, visual_pos)
        v_visual = visual_memory_v

        visual_info, _ = self.visual_attn(query=q_visual, key=k_visual, value=v_visual,
                                          key_padding_mask=visual_memory_key_padding_mask)
                                          # attn_mask=None

        # --- First Residual Connection & Normalization ---
        # Add the visual info back to the *original* query (like VLTVG)
        # Followed by dropout and layer normalization (Post-Residual Norm)
        query = self.norm2(query + self.dropout2(visual_info))

        # 3. --- Feed Forward Network ---
        # Pass the result through the FFN
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(query))))

        # --- Second Residual Connection & Normalization ---
        # Add FFN output back to its input
        # Followed by dropout and layer normalization (Post-FFN Norm)
        query = self.norm3(query + self.dropout3(ffn_output))

        return query

class TransformerDecoder(nn.Module):
    """
    Generic Transformer Decoder wrapper that stacks layers.
    Modified to handle separate text/visual inputs for VltvgStyleDecoderLayer.
    """
    def __init__(self, decoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        # Final normalization after all layers (optional, but common)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                # --- Query Inputs ---
                tgt: Tensor,                          # The input query sequence
                query_pos: Optional[Tensor] = None,   # Positional encoding for the query

                # --- Text Memory Inputs ---
                text_memory: Optional[Tensor] = None, # Encoded text features
                text_memory_key_padding_mask: Optional[Tensor] = None, # Padding mask for text
                text_pos: Optional[Tensor] = None,    # Positional encoding for text

                # --- Visual Memory Inputs ---
                visual_memory: Optional[Tensor] = None, # Encoded visual features
                visual_memory_key_padding_mask: Optional[Tensor] = None, # Padding mask for visual
                visual_pos: Optional[Tensor] = None,    # Positional encoding for visual

                # --- Control ---
                layer_idx: Optional[int] = None       # Optional: compute only a specific layer
               ):
        """
        Forward pass through the stack of decoder layers.
        Passes the separated text/visual arguments to each layer.
        """
        output = tgt

        if layer_idx is None:
            # Compute all layers
            output_list = []
            for layer in self.layers:
                # Pass all arguments to the VltvgStyleDecoderLayer's forward method
                output = layer(
                    query=output, # 'output' from previous layer becomes the query for the next
                    query_pos=query_pos,
                    text_memory=text_memory,
                    text_memory_key_padding_mask=text_memory_key_padding_mask,
                    text_pos=text_pos,
                    visual_memory=visual_memory,
                    visual_memory_key_padding_mask=visual_memory_key_padding_mask,
                    visual_pos=visual_pos,
                    # text_informed_query_pos is not explicitly passed here,
                    # layer will use its default behavior (likely reusing query_pos)
                )
                # Apply final norm *after* each layer's output before collecting
                # (This matches DETR/Deformable DETR style of intermediate outputs)
                # If you only want the final output after all layers, move norm outside loop
                output_list.append(self.norm(output))

            # Stack intermediate outputs if needed (e.g., for auxiliary losses)
            return torch.stack(output_list, dim=0)
        else:
            # Compute only a specific layer
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise IndexError(f"layer_idx {layer_idx} out of range for {self.num_layers} layers")
            layer = self.layers[layer_idx]
            output = layer(
                query=output,
                query_pos=query_pos,
                text_memory=text_memory,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                text_pos=text_pos,
                visual_memory=visual_memory,
                visual_memory_key_padding_mask=visual_memory_key_padding_mask,
                visual_pos=visual_pos,
            )
            # Apply final norm to the single layer's output
            return self.norm(output)


