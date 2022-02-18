import argparse
import os
import numpy as np


# Training
from trainer import trainer_synapse


# Transunet
from segmentation_models.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from segmentation_models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# Unet
from segmentation_models.unet import create_model


# MedicalTransformer
from segmentation_models.medt import gated

# Arguments for implementation of the model

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse', help='root dir for data') 
parser.add_argument('--list_dir', type=str,
                    default='./data/lists_Synapse', help='list dir')
parser.add_argument('--model_name', type=str,
                    default='TransUnet', help='select one model among: Unet / TransUnet / MedicalTransformer / Segmenter')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network') 
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train') 
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train') 
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--in_channels', type=int,
                    default=1, help='input channel size of network input')
parser.add_argument('--is_pretrained', action='store_true', help='pretrained model or not')

# Arguments for normalization

parser.add_argument('--apply_normalization', action='store_true', help='normalization of your dataset (you have to know the mean and the std of your dataset)')
parser.add_argument('--mean', type=int,
                    default=89.67757, help='mean of your dataset')
parser.add_argument('--std', type=bool,
                    default=173.3126, help='std of your dataset')


args = parser.parse_args()


if __name__ == "__main__":
    snapshot_path = "./model_saved/{}/".format(args.model_name)
    snapshot_path = snapshot_path + 'epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrained else snapshot_path

    # TransUnet
    if args.model_name == 'TransUnet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = 3
        vit_patches_size = 16
        config_vit.patches.grid = (int(args.img_size / vit_patches_size), int(args.img_size / vit_patches_size))

        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        
        if args.is_pretrained:
            pretrained_path='./segmentation_models/transunet/vit_checkpoint/R50-ViT-B_16.npz'
            net.load_from(weights=np.load(pretrained_path)) 

    # Unet
    if args.model_name == 'Unet':
        if args.is_pretrained:
            weights_unet='imagenet'
        else:
            weights_unet=None

        net = create_model('Unet',
            encoder_name='resnet34',
            encoder_depth=5,
            encoder_weights=weights_unet,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels=1,
            classes=5,
            activation=None,
            aux_params=None).cuda()
    
    if args.model_name == 'MedicalTransformer':
        net = gated(img_size=args.img_size, imgchan=args.in_channels, num_classes=args.num_classes)
    
    if args.model_name == 'Segmenter':
        pass
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    # Training
    trainer_synapse(args, net, snapshot_path)
