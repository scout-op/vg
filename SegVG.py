import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer, Triple_Alignment
# from .visual_model.transformer import VisionLanguageDecoder
from .visual_model.lightweight_transformer import VisionLanguageDecoder


class SegVG(nn.Module):
    def __init__(self, args):
        super(SegVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        ## Extraction ##
        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)
        self.N = 1 + 5
        self.reg_seg_src = nn.Embedding(self.N, hidden_dim)
        self.reg_seg_pos = nn.Embedding(self.N, hidden_dim)
        self.deepfuse = Triple_Alignment(num_layers=6, embed_dim=hidden_dim, num_heads=8) # this is the triple alignment module, somehow I named it deepfuse

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)
        self.object_proj = nn.Linear(hidden_dim, hidden_dim)

        ## Encoder ##
        num_total = self.num_visu_token + self.num_text_token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.vl_transformer = build_vl_transformer(args) # this is the encoder module, somehow I named it vl_transformer

        ## Decoder ##
        # self.vl_decoder = VisionLanguageDecoder(num_decoder_layers=6, d_model=hidden_dim, nhead=8) # this is the decoder module, somehow I named it vl_decoder
        ## Decoder ##
        # 检查是否使用轻量化解码器
        # use_lightweight_decoder = getattr(args, 'use_lightweight_decoder', True)
       
            # 使用轻量化解码器
        # 保持不变，或者按需添加参数
        self.vl_decoder = VisionLanguageDecoder(
            num_decoder_layers=6, # Use args if available
            d_model=hidden_dim,
            nhead=args.vl_nheads, # Use args if available
            dim_feedforward=2048, # Use args if available
            dropout=args.dropout, # Use args if available
            # activation="relu", # Assuming default or set in VisionLanguageDecoder
            img_feat_chunk_num=1 # Set if needed
        )
        # print("Using lightweight decoder with 3 layers and parameter sharing")
   
           

        ## Prediction ##
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.seg_embed = MLP(hidden_dim*2, hidden_dim, 1, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        ## Extraction ## 
        # visual resnet encode #
        visual_src, visual_mask, visual_pos = self.visumodel(samples=img_data)

        # language encode #
        text_src, extended_attention_mask, head_mask = self.textmodel(input_ids=text_data.tensors, attention_mask=text_data.mask)
        text_mask = ~text_data.mask.to(torch.bool)

        # triple alignment #
        tgt_src = self.reg_seg_src.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_pos = self.reg_seg_pos.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(0, 12, 2):
            # detr
            visual_src, visual_mask, visual_pos = self.visumodel(src=visual_src, mask = visual_mask, pos=visual_pos, layer_idx=i // 2)
            # bert1
            text_src = self.textmodel(input_hidden_states=text_src, attention_mask=extended_attention_mask, head_mask=head_mask[i], layer_idx=i)
            # bert2
            text_src = self.textmodel(input_hidden_states=text_src, attention_mask=extended_attention_mask, head_mask=head_mask[i+1], layer_idx=i+1)
            # triple_align
            visual_src, text_src, tgt_src = self.deepfuse(
                tgt_src = tgt_src, tgt_pos = tgt_pos,
                visual_src=visual_src, visual_mask = visual_mask, visual_pos = visual_pos, 
                text_src=text_src.permute(1, 0, 2), text_mask = text_mask, layer_idx=i // 2)
            text_src = text_src.permute(1, 0, 2) # bert format

        visual_src = self.visu_proj(visual_src)
        text_src = self.text_proj(text_src) 
        text_src = text_src.permute(1, 0, 2) # permute BxLenxC to LenxBxC
        text_mask = text_mask.flatten(1)
        tgt_src = self.object_proj(tgt_src)

        ## Encoder ##
        vl_src = torch.cat([text_src, visual_src], dim=0)
        vl_mask = torch.cat([text_mask, visual_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (L + V)xBxC

        ## == Decoder Stage (vl_decoder - MODIFIED CALL) == ##
        # --- Split the encoder output and corresponding inputs for the VLTVG-style decoder ---
        # Use self.num_text_token which was stored in __init__

        # Split vg_hs (Encoder output features)
        text_memory_from_encoder = vg_hs[:self.num_text_token]        # (L, B, C)
        visual_memory_from_encoder = vg_hs[self.num_text_token:]      # (V, B, C)

        # Split vl_mask (Padding mask used in encoder)
        # Note: vl_mask has shape (B, L+V)
        text_memory_key_padding_mask = vl_mask[:, :self.num_text_token] # (B, L)
        visual_memory_key_padding_mask = vl_mask[:, self.num_text_token:]# (B, V)

        # Split vl_pos (Positional encoding used in encoder input)
        # Note: vl_pos has shape (L+V, B, C)
        text_pos_from_encoder_input = vl_pos[:self.num_text_token]      # (L, B, C)
        visual_pos_from_encoder_input = vl_pos[self.num_text_token:]    # (V, B, C)

        # --- Call the modified decoder with separated inputs ---
        # Output shape depends on TransformerDecoder impl: (Layers, N, B, C) or (N, B, C) if layer_idx is used
        decoder_output = self.vl_decoder(
            tgt=tgt_src,                                 # Query (N, B, C)
            query_pos=tgt_pos,                           # Query pos embedding (N, B, C)

            text_memory=text_memory_from_encoder,        # Text features from encoder (L, B, C)
            text_memory_key_padding_mask=text_memory_key_padding_mask, # Mask (B, L)
            text_pos=text_pos_from_encoder_input,        # Positional embedding (L, B, C)

            visual_memory=visual_memory_from_encoder,    # Visual features from encoder (V, B, C)
            visual_memory_key_padding_mask=visual_memory_key_padding_mask, # Mask (B, V)
            visual_pos=visual_pos_from_encoder_input     # Positional embedding (V, B, C)
            # layer_idx=None # Compute all layers, output shape (Layers, N, B, C)
        )
        # Assuming output is (Layers, N, B, C) based on your previous TransformerDecoder code
        # Use the output from the *last* decoder layer for predictions
        tgt_src_final_layer = decoder_output[-1] # Shape: (N, B, C)

        pred_box = self.bbox_embed(decoder_output[:, 0]).sigmoid() # 输入 (Layers, B, C) -> 输出 (Layers, B, 4)



        # pred_box = self.bbox_embed(tgt_src[:, 0]).sigmoid()

        if self.training:
            # --- 获取所有层的分割查询输出 ---
            # decoder_output shape: (Layers, N, B, C)
            # 选择除了第一个查询（用于bbox）之外的所有查询 (1:)，跨所有层 (:)
            seg_queries_all_layers = decoder_output[:, 1:] # Shape: (Layers, N_s, B, C)
            Layers, N_s, B, C = seg_queries_all_layers.shape # 获取维度 L, N_s, B, C

            # --- 获取编码器的视觉特征 (这部分和之前一样) ---
            # visual_features_for_seg = vg_hs[self.num_text_token:] # (V, B, C)
            # 或者用 vg_hs[-400:] 如果确定是这个值
            visua = vg_hs[self.num_text_token:] # Shape: (V, B, C)
            V = visua.shape[0]

            # --- 扩展查询和视觉特征以进行组合 ---
            # 目标：将每层的查询 (L, N_s, B, C) 与 视觉特征 (V, B, C) 结合
            # 扩展查询：(L, N_s, B, C) -> (L, N_s, 1, B, C) -> (L, N_s, V, B, C)
            expanded_queries = seg_queries_all_layers.unsqueeze(2).expand(Layers, N_s, V, B, C)

            # 扩展视觉特征：(V, B, C) -> (1, 1, V, B, C) -> (L, N_s, V, B, C)
            expanded_visual = visua.unsqueeze(0).unsqueeze(0).expand(Layers, N_s, V, B, C)

            # --- 拼接 ---
            visua_combined_all_layers = torch.cat([expanded_queries, expanded_visual], dim=-1) # Shape: (L, N_s, V, B, C*2)

            # --- 通过 MLP 进行预测 ---
            # 输入 (L, N_s, V, B, C*2) -> 输出 (L, N_s, V, B, 1) -> (L, N_s, V, B)
            pred_mask_logits = self.seg_embed(visua_combined_all_layers)[..., 0]

            # --- 调整维度以匹配损失函数的期望 (L, N, bs, V) ---
            # 当前形状: (L, N_s, V, B)
            # 期望形状: (L, N_s, B, V)  (对应 L, N, bs, _)
            # 需要交换最后两个维度 V 和 B
            pred_mask_list = pred_mask_logits.permute(0, 1, 3, 2) # Shape: (L, N_s, B, V)

            return pred_box, pred_mask_list # 现在 pred_mask_list 是 4D 的

        else: # 推理模式 (也可能需要调整，但先修复训练模式)
            # 推理模式通常只用最后一层，需要确认是否也要修改
            visua = vg_hs[self.num_text_token:] # Or vg_hs[-400:]
            V,B,C = visua.shape

            # 使用最后一层的输出
            tgt_src_final_layer = decoder_output[-1] # Shape: (N, B, C)

            seg_index = 1 # 通常用第一个分割查询
            seg_query_final = tgt_src_final_layer[seg_index] # Shape (B, C)

            # 扩展查询和视觉特征
            expanded_query = seg_query_final.unsqueeze(0).expand(V, B, C) # (V, B, C)
            visua_combined = torch.cat([ expanded_query, visua ], dim=-1) # (V, B, C*2)

            # 预测并调整形状
            seg_output = self.seg_embed(visua_combined)[:,:,0].transpose(0, 1) # (V, B) -> (B, V)

            # 注意：这里的 pred_box 仍然是 (Layers, B, 4)，推理时通常也只取最后一层
            # pred_box_final = pred_box[-1] # Shape: (B, 4)

            # return pred_box_final, seg_output.sigmoid()
            # 或者如果推理也需要所有层的bbox？取决于评估代码
            return pred_box, seg_output.sigmoid() # 暂时保持 pred_box 为 4D

        # # Predict segmentation masks using other queries and visual features from encoder
        # if self.training:
        #     # Use visual features directly from encoder output `vg_hs`
        #     visual_features_for_seg = vg_hs[self.num_text_token:] # (V, B, C)
        #     V, B, C = visual_features_for_seg.shape
        #     # Use segmentation queries output from the last decoder layer
        #     seg_queries_output = last_layer_output[1:] # Shape (N_s, B, C), where N_s = N - 1
        #     N_s = seg_queries_output.shape[0]
        #     assert N_s == self.N - 1

        #     # Expand queries and visual features for pairwise interaction
        #     # seg_queries_output: (N_s, B, C) -> (N_s, 1, B, C) -> (N_s, V, B, C)
        #     expanded_queries = seg_queries_output.unsqueeze(1).expand(N_s, V, B, C)
        #     # visual_features_for_seg: (V, B, C) -> (1, V, B, C) -> (N_s, V, B, C)
        #     expanded_visual = visual_features_for_seg.unsqueeze(0).expand(N_s, V, B, C)

        #     # Concatenate for MLP input
        #     # (N_s, V, B, C*2)
        #     seg_mlp_input = torch.cat([expanded_queries, expanded_visual], dim=-1)

        #     # Predict masks (N_s, V, B, 1) -> (N_s, V, B)
        #     pred_mask_logits = self.seg_embed(seg_mlp_input)[..., 0]
        #     # Reshape to (N_s, B, V) for standard mask prediction format
        #     pred_mask_logits = pred_mask_logits.permute(0, 2, 1) # (N_s, B, V)

        #     # Stack layer outputs if needed for auxiliary losses (use decoder_output directly)
        #     # Example: Assuming bbox prediction uses query 0 from each layer
        #     # aux_pred_boxes = self.bbox_embed(decoder_output[:, 0]).sigmoid() # (Layers, B, 4)
        #     # Similar logic can be applied for auxiliary masks if needed

        #     # Return last layer predictions (and potentially auxiliary outputs)
        #     # Adjust return based on your loss calculation needs
        #     return pred_box, pred_mask_logits # Return logits for BCEWithLogitsLoss usually

        # else: # Inference
        #     # Use visual features directly from encoder output
        #     visual_features_for_seg = vg_hs[self.num_text_token:] # (V, B, C)
        #     V, B, C = visual_features_for_seg.shape

        #     # Use the segmentation query output (e.g., the first seg query, index 1) from the last layer
        #     seg_query_output = last_layer_output[1] # Shape (B, C)

        #     # Expand query and visual features
        #     # seg_query_output: (B, C) -> (1, B, C) -> (V, B, C)
        #     expanded_query = seg_query_output.unsqueeze(0).expand(V, B, C)
        #     # visual_features_for_seg: (V, B, C)

        #     # Concatenate for MLP input
        #     seg_mlp_input = torch.cat([expanded_query, visual_features_for_seg], dim=-1) # (V, B, C*2)

        #     # Predict mask logits (V, B, 1) -> (V, B)
        #     seg_output_logits = self.seg_embed(seg_mlp_input)[..., 0]
        #     # Transpose to (B, V)
        #     seg_output_logits = seg_output_logits.transpose(0, 1)

        #     return pred_box, seg_output_logits.sigmoid() # Return sigmoid for evaluation

        # if self.training:
        #     visua = vg_hs[-400:]
        #     V,B,C = visua.shape
        #     L = tgt_src.shape[0]
        #     N_s = self.N - 1
        #     visua = visua.view(1,1,V,B,C).expand(L,N_s,V,B,C)
        #     # L,N,1,B,C & L,N,V,B,C
        #     visua = torch.cat([
        #         tgt_src[:, -N_s:].view(L,N_s,1,B,C).expand(L,N_s,V,B,C), 
        #         visua], dim=-1)
        #     # L,N,V,B,C --> L,N,V,B,1 --> L,N,V,B --> L,N,B,V
        #     pred_mask_list = self.seg_embed(visua)[:,:,:,:,0].permute(0,1,3,2)

        #     return pred_box, pred_mask_list
        # else:
        #     # confidence score
        #     visua = vg_hs[-400:]
        #     V,B,C = visua.shape
            
        #     seg_index = 1
        #     visua = torch.cat([
        #         tgt_src[-1, [seg_index]].expand(V,B,C), 
        #         visua
        #         ], dim=-1)
        #     seg_output = self.seg_embed(visua)[:,:,0].transpose(0, 1)

        #     return pred_box, seg_output.sigmoid()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x