import ml_collections

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""

    config = ml_collections.ConfigDict()

    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768   
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.patches.grid = (16, 16)

    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config







