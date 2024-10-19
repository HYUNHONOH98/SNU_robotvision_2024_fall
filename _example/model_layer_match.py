import torch

def map_mask2former_weight(model, device):
  mapped_weights= {}

  pretrained_weights = torch.load("/home/hyunho/sfda/swin_tiny_patch4_window7_224.pth", map_location= device)['model']
  # Example mappings for patch embedding layers
  mapped_weights[f'model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.weight'] = pretrained_weights['patch_embed.proj.weight']
  mapped_weights[f'model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.bias'] = pretrained_weights['patch_embed.proj.bias']
  mapped_weights[f'model.pixel_level_module.encoder.embeddings.norm.weight'] = pretrained_weights['patch_embed.norm.weight']
  mapped_weights[f'model.pixel_level_module.encoder.embeddings.norm.bias'] = pretrained_weights['patch_embed.norm.bias']

  layers = [[(0,0), (0,1)],
            [(1,0), (1,1)],
            [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5)],
            [(3,0), (3,1)]]

  # Example mappings for encoder layers (layer 0, block 0)
  for layer in layers:
    for layer_idx, block_idx in layer:
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.norm1.weight"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.norm1.bias"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index"]
      weight_chunks = torch.chunk(pretrained_weights[f'layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight'], chunks=3, dim=0)
      bias_chunks = torch.chunk(pretrained_weights[f'layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias'], chunks=3, dim=0)
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.query.weight'] = weight_chunks[0]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.query.bias'] = bias_chunks[0]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.key.weight'] = weight_chunks[1]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.key.bias'] = bias_chunks[1]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.value.weight'] = weight_chunks[2]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attn.value.bias'] = bias_chunks[2]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight'] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight"]
      mapped_weights[f'model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias'] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.norm2.weight"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.norm2.bias"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias"] = pretrained_weights[f"layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias"]
    if layer_idx in [0,1,2]:
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.downsample.reduction.weight"] = pretrained_weights[f"layers.{layer_idx}.downsample.reduction.weight"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.downsample.norm.weight"] = pretrained_weights[f"layers.{layer_idx}.downsample.norm.weight"]
      mapped_weights[f"model.pixel_level_module.encoder.encoder.layers.{layer_idx}.downsample.norm.bias"] = pretrained_weights[f"layers.{layer_idx}.downsample.norm.bias"]

  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage4.weight"] = pretrained_weights["norm.weight"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage4.bias"] = pretrained_weights["norm.bias"]

  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage1.weight"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage1.bias"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage2.weight"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage2.bias"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage3.weight"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage3.bias"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage4.weight"]
  # mapped_weights[f"model.pixel_level_module.encoder.hidden_states_norms.stage4.bias"]

  model.load_state_dict(mapped_weights, strict=False)

  return model