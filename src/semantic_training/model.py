import torch
import torch.nn as nn
import math
from parameter import *
if not train_mode:
    from test_parameter import *
# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        # self.tanh_clipping = 10  # [Deleted] 彻底移除
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None, return_attention_weights=False):
        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        
        # [关键修复] 显式处理 Mask
        if mask is not None:
            # 1. 确保 mask 是布尔类型 (防止 int64/float32 比较失败)
            if mask.dtype != torch.bool:
                # 假设 mask 中 1 是无效，0 是有效
                bool_mask = mask > 0.5 
            else:
                bool_mask = mask
            
            # 2. 打印调试信息 (仅在第一次运行时，或者你怀疑出错时打开)
            # if torch.rand(1).item() < 0.01: # 随机抽样打印
            #     print(f"[Attention DEBUG] U shape: {U.shape}, Mask shape: {bool_mask.shape}")
            #     print(f"[Attention DEBUG] Mask sum: {bool_mask.sum().item()}")

            # 3. 应用极小的负数
            U = U.masked_fill(bool_mask, -1e9)

        # 保存原始注意力权重用于可视化
        attention_weights = torch.softmax(U, dim=-1) if return_attention_weights else None
        attention = torch.log_softmax(U, dim=-1)

        if return_attention_weights:
            return attention, attention_weights
        return attention

# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PolicyNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PolicyNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        # self.r_r_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        
        # self.prev_action_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.pointer = SingleHeadAttention(embedding_dim)
        
        # Heading Embedding (36 bins -> embedding_dim)
        self.heading_embedding = nn.Embedding(NUM_ANGLES_BIN, embedding_dim)
        
        # Fusion layer to combine neighbor feature and heading feature
        self.feature_fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Removed orientation_head

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, utility_mask):
        node_feature = self.initial_embedding(node_inputs)
        # print("mask",node_feature.shape, node_padding_mask.shape)
        enhanced_node_feature = self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return enhanced_node_feature

    def output_policy(self, enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask, greedy=False, return_attention_weights=False, neighbor_best_headings=None):
        
        current_edge = edge_inputs.permute(0, 2, 1)
        
        embedding_dim = enhanced_node_feature.size()[2]
        # print("current_edge", current_edge.size())
        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))

        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))

        if edge_padding_mask is not None:
            current_mask = edge_padding_mask

        else:
            current_mask = None

        # if not ALLOW_STAY:
        #     current_mask[:, :, 0] = 1  # don't stay at current position

        enhanced_current_node_feature, decoder_attention = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        enhanced_current_node_feature = self.current_embedding(torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1))
        
        # --- New Logic for Relative Selection ---
        if neighbor_best_headings is None:
             # Should not happen in training, but for safety
             raise ValueError("neighbor_best_headings required for output_policy")

        batch_size, k_size, num_candidates = neighbor_best_headings.size()
        
        # Embed headings: (Batch, K, 3, Dim)
        heading_features = self.heading_embedding(neighbor_best_headings)
        
        # Expand neighbor features: (Batch, K, 3, Dim)
        neigboring_feature_expanded = neigboring_feature.unsqueeze(2).repeat(1, 1, num_candidates, 1)
        
        # Combine: (Batch, K, 3, 2*Dim)
        combined = torch.cat((neigboring_feature_expanded, heading_features), dim=-1)
        
        # Fuse: (Batch, K, 3, Dim)
        fused_features = self.feature_fusion(combined)
        
        # Flatten to (Batch, K*3, Dim) for Pointer Attention
        fused_features_flat = fused_features.view(batch_size, k_size * num_candidates, embedding_dim)
        
        # Expand Mask: (Batch, 1, K) -> (Batch, 1, K*3)
        if current_mask is not None:
            # current_mask is (Batch, 1, K). 1 means masked (padding).
            current_mask_expanded = current_mask.unsqueeze(-1).repeat(1, 1, 1, num_candidates).view(batch_size, 1, k_size * num_candidates)
        else:
            current_mask_expanded = None

        logp = self.pointer(enhanced_current_node_feature, fused_features_flat, current_mask_expanded)
        logp = logp.squeeze(1) 
        return logp



    def forward(self, node_inputs, edge_inputs, current_index, node_padding_mask=None, edge_padding_mask=None, edge_mask=None, utility_mask=None, neighbor_best_headings=None, greedy=False):

# # --- DEBUG START: 完整打印所有输入 ---
#         # 设置打印选项：不折叠(threshold=inf)，行宽设大一点避免换行过多(linewidth=2000)
#         import sys
#         torch.set_printoptions(profile="full", linewidth=2000, threshold=float('inf'))
        
#         print("\n" + "="*50)
#         print(">>> PolicyNet Forward Input Debug <<<")
#         print("="*50)

#         # 1. node_inputs
#         print(f"\n[node_inputs] Shape: {node_inputs.shape}")
#         print(node_inputs)

#         # 2. edge_inputs
        # print(f"\n[edge_inputs] Shape: {edge_inputs.shape}")
        # print(edge_inputs)

#         # 3. current_index
#         print(f"\n[current_index] Shape: {current_index.shape}")
#         print(current_index)

#         # 4. node_padding_mask
#         if node_padding_mask is not None:
#             print(f"\n[node_padding_mask] Shape: {node_padding_mask.shape}")
#             print(node_padding_mask)
#         else:
#             print("\n[node_padding_mask] is None")

#         # 5. edge_padding_mask
        # if edge_padding_mask is not None:
        #     print(f"\n[edge_padding_mask] Shape: {edge_padding_mask.shape}")
        #     print(edge_padding_mask)
        # else:
        #     print("\n[edge_padding_mask] is None")

#         # 6. edge_mask (Attention Mask)
#         if edge_mask is not None:
#             print(f"\n[edge_mask] Shape: {edge_mask.shape}")
#             print(edge_mask)
#         else:
#             print("\n[edge_mask] is None")

#         # 7. utility_mask
#         if utility_mask is not None:
#             print(f"\n[utility_mask] Shape: {utility_mask.shape}")
#             print(utility_mask)
#         else:
#             print("\n[utility_mask] is None")

#         # 8. neighbor_best_headings (重要特征)
#         if neighbor_best_headings is not None:
#             print(f"\n[neighbor_best_headings] Shape: {neighbor_best_headings.shape}")
#             print(neighbor_best_headings)
#         else:
#             print("\n[neighbor_best_headings] is None")

#         # 9. Booleans
#         print(f"\n[Flags] greedy: {greedy}, return_attention_weights: {return_attention_weights}")
        
#         print("="*50 + "\n")
        
#         # 记得把打印设置还原，否则控制台后面可能会被刷屏
#         # torch.set_printoptions(profile="default") 
#         # --- DEBUG END ---
#         assert 0
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask, utility_mask)
        
        logp = self.output_policy(enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask, greedy, neighbor_best_headings=neighbor_best_headings)
        return logp


class QNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(QNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.action_embedding = nn.Linear(embedding_dim*3, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim, 1) # Output 1 Q-value per action
        
        # Heading Embedding
        self.heading_embedding = nn.Embedding(NUM_ANGLES_BIN, embedding_dim)
        self.feature_fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, utility_mask):
        embedding_feature = self.initial_embedding(node_inputs)
        embedding_feature = self.encoder(src=embedding_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)
        return embedding_feature

    def output_q_values(self, enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask, neighbor_best_headings=None):
        # k_size = edge_inputs.size()[2] // N_ROBOTS
        k_size = edge_inputs.size()[2]
        current_edge = edge_inputs
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]
        # print('current_edge', current_edge.size())
        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        # print(current_index.size())
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim)) #(batch_size, 2, embedding_dim)
        # print('enhanced_current_node_feature', enhanced_node_feature.size(), 'current_node_feature', current_node_feature.size())
        enhanced_current_node_feature, attention_weights = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        # print('enhanced_current_node_feature', enhanced_current_node_feature.size(), 'current_node_feature', current_node_feature.size(), 'neigboring_feature', neigboring_feature.size())
       
        if edge_padding_mask is not None:
            current_mask = edge_padding_mask
        else:
            current_mask = None
        # if not ALLOW_STAY:
        #     current_mask[:, :, 0] = 1  # don't stay at current position

        # --- New Logic ---
        if neighbor_best_headings is None:
             raise ValueError("neighbor_best_headings required for output_q_values")
        
        batch_size, _, num_candidates = neighbor_best_headings.size()
        
        # Embed headings: (Batch, K, 3, Dim)
        heading_features = self.heading_embedding(neighbor_best_headings)
        
        # Expand neighbor features: (Batch, K, 3, Dim)
        neigboring_feature_expanded = neigboring_feature.unsqueeze(2).repeat(1, 1, num_candidates, 1)
        
        # Combine and Fuse: (Batch, K, 3, Dim)
        fused_features = self.feature_fusion(torch.cat((neigboring_feature_expanded, heading_features), dim=-1))
        
        # Expand context features
        enhanced_current_node_feature_expanded = enhanced_current_node_feature.repeat(1, k_size, 1).unsqueeze(2).repeat(1, 1, num_candidates, 1)
        current_node_feature_expanded = current_node_feature.repeat(1, k_size, 1).unsqueeze(2).repeat(1, 1, num_candidates, 1)
        
        # Concatenate for Action Features: (Batch, K, 3, 3*Dim)
        action_features = torch.cat((enhanced_current_node_feature_expanded, current_node_feature_expanded, fused_features), dim=-1)
        
        action_features = self.action_embedding(action_features)
        q_values = self.q_values_layer(action_features) # (Batch, K, 3, 1)
        
        q_values = q_values.view(batch_size, k_size * num_candidates) # Flatten
        
        # Mask
        if current_mask is not None:
            current_mask = current_mask.permute(0, 2, 1) # (Batch, 1, K) -> (Batch, K, 1)? 
            # Wait, edge_padding_mask in get_observations is (1, 1, K).
            # Here current_mask = edge_padding_mask.
            # permute(0, 2, 1) -> (1, K, 1).
            # We want (Batch, K*3).
            # Expand (Batch, K, 1) to (Batch, K, 3) then flatten.
            current_mask_expanded = current_mask.repeat(1, 1, num_candidates).view(batch_size, k_size * num_candidates)
            
            zero = torch.zeros_like(q_values).to(q_values.device)
            q_values = torch.where(current_mask_expanded == 1, zero, q_values)

        return q_values, attention_weights

    def forward(self, node_inputs, edge_inputs, current_index, node_padding_mask=None, edge_padding_mask=None,
                edge_mask=None, utility_mask=None, neighbor_best_headings=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask, utility_mask)
        q_values, attention_weights = self.output_q_values(enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask, neighbor_best_headings=neighbor_best_headings)
        return q_values, attention_weights

