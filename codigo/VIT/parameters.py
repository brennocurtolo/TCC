# PRIMEIRA VERSÃO (CPU) 

# learning_rate = 0.001
# weight_decay = 0.0001
# batch_size = 128
# num_epochs = 50  # For real training, use num_epochs=100. 10 is a test value
# image_size = 72  # We'll resize input images to this size
# patch_size = 6  # Size of the patches to be extract from the input images
# num_patches = (image_size // patch_size) ** 2
# projection_dim = 64
# num_heads = 4
# transformer_units = [
#     projection_dim * 2,
#     projection_dim,
# ]  # Size of the transformer layers
# transformer_layers = 8
# mlp_head_units = [
#     2048,
#     1024,
# ]  # Size of the dense layers of the final classifier


learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 20  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 10
mlp_head_units = [
    3072, 
    2048,
]  # Size of the dense layers of the final classifier
