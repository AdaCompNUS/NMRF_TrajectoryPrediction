import torch
import torch.nn as nn

class st_encoder_with_attn(nn.Module):
	def __init__(self, n_layer=1):
		super().__init__()
		channel_in = 6
		channel_out = 32
		dim_kernel = 3
		dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, dim_embedding_key, n_layer, batch_first=True)

		self.self_attn = nn.MultiheadAttention(embed_dim=dim_embedding_key, num_heads=2)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X, mask):
		'''
		X: b, T, 6
		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		# state_x = state_x.squeeze(0)
		state_x = state_x[-1].unsqueeze(1)    # take out last layer if multiple layers

		# # n_samples, 1, dim_gru_embed: (seq_len, B, embed_dim)
		# # attn_mask: (tar_seq_len, src_seq_len)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		state_x = self.self_attn(state_x, state_x, state_x, attn_mask=mask)[0]   # attn_output, attn_weights

		return state_x.squeeze(1)


class MLP(nn.Module):
	def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)
			if i < len(self.layers) - 1:    # only for hidden layers
				x = self.activation(x)
				x = self.dropout(x)
		return x