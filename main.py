from mode import *
import argparse


parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

parser.add_argument("--LR_path", type = str, default = '../DIV2K/DIV2K_train_LR_bicubic/X4')#Low-resolution image directory path.
parser.add_argument("--GT_path", type = str, default = '../DIV2K/DIV2K_train_HR/')#Ground truth high-resolution image directory path.
parser.add_argument("--res_num", type = int, default = 5)#Number of residual blocks in model.
parser.add_argument("--num_workers", type = int, default = 0)#Number of subprocesses for data loading.
parser.add_argument("--batch_size", type = int, default = 7)#Batch size for training or evaluation.
parser.add_argument("--L2_coeff", type = float, default = 1.0)#Coefficient for L2 loss function.
parser.add_argument("--adv_coeff", type = float, default = 1e-3)#Coefficient for adversarial loss function.
parser.add_argument("--tv_loss_coeff", type = float, default = 0.0)#Coefficient for total variation loss function.
parser.add_argument("--pre_train_epoch", type = int, default = 10)#Number of epochs for pre-training.
parser.add_argument("--fine_train_epoch", type = int, default = 5)#Number of epochs for fine-tuning.
parser.add_argument("--scale", type = int, default = 4)#Scale factor for super-resolution.
parser.add_argument("--patch_size", type = int, default = 24)#Size of image patches used during training
parser.add_argument("--feat_layer", type = str, default = 'relu5_4')#Layer of pre-trained VGG network for feature extraction.
parser.add_argument("--vgg_rescale_coeff", type = float, default = 0.006)#Coefficient for rescaling VGG perceptual loss.
parser.add_argument("--fine_tuning", type = str2bool, default = False)#Enable or disable fine-tuning.
parser.add_argument("--in_memory", type = str2bool, default = True)#Load entire dataset into memory or not.
parser.add_argument("--generator_path", type = str)#Path to pre-trained generator model.
parser.add_argument("--mode", type = str, default = 'train')#Mode of operation (e.g., 'train', 'eval', etc.).

args = parser.parse_args()

if args.mode == 'train':
    train(args)
    
elif args.mode == 'test':
    test(args)
    
elif args.mode == 'test_only':
    test_only(args)

