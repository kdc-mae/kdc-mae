# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
from traintest_cavmae import train

# pretrain cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt", "kinetics"])
parser.add_argument("--dataset_mean", type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments, only for preliminary experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=None)
parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
parser.add_argument("--mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])

parser.add_argument("--four_tiling", type=ast.literal_eval, help="enable 2x2 tiling in the dataloader")
parser.add_argument("--restrict_randomness", type=ast.literal_eval, help="restrict randomness of patches using static numpy files")

parser.add_argument("--model_type", type=str, default='vanilla', help="CAV model used", choices=['vanilla', 'latent'])
parser.add_argument("--dynamic_weight_normalization_method", type=str, default='unormalized', help="Dynamic Weight Normalization Method", choices=['total_sum1', 'individual_sum1', 'unormalized'])
parser.add_argument("--absolute_noise", type=ast.literal_eval, help="Absolute Noise")
parser.add_argument("--dynamic_weighting", type=ast.literal_eval, help="Dynamic Weighting")
parser.add_argument("--knowledge_distillation", type=ast.literal_eval, help="Knowledge Distillation")
parser.add_argument("--k_value", type=float, default=-0.25, help="Value of k")
parser.add_argument("--split_decoder", type=ast.literal_eval, help="Split Decoder")
parser.add_argument("--dual_mask", type=ast.literal_eval, help="Dual Mask")
parser.add_argument("--complementary", type=ast.literal_eval, help="Complimentary")
parser.add_argument("--AAVV", type=ast.literal_eval, help="Audio Audio and Video Video Dual Mask KL div")
parser.add_argument("--resume_from_checkpoint", type=str, help="Checkpoint path to resume from", default="", required=False)
parser.add_argument("--triple_mask", type=ast.literal_eval, help="Triple Mask")
parser.add_argument("--kd_weight", type=int, default=10, help="Kd weight")
args = parser.parse_args()

print('current model type is: ', args.model_type)
print('current dynamic weighting is: ', args.dynamic_weighting)
print('current dynamic weighting normalization method is: ', args.dynamic_weight_normalization_method)
print('current knowledge distillation is: ', args.knowledge_distillation)
print('current k value is: ', args.k_value)
print('current absolute noise is: ', args.absolute_noise)
print('current split decoder is: ', args.split_decoder)
print('current dual mask is: ', args.dual_mask)
print('current complementary is: ', args.complementary)

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        # samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        weight_path = '/home/amukherjee/Static/train_filtered_vgg_full_cleaned_weight.csv'
        samples_weight = np.loadtxt(weight_path, delimiter=',')
    else:
        # samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
        if args.dataset == 'vggsound':
            weight_path_else = '/home/amukherjee/Static/train_filtered_vgg_full_cleaned_weight.csv'
            samples_weight = np.loadtxt(weight_path_else, delimiter=',')
        elif args.dataset == 'audioset':
            weight_path_else = '/home/amukherjee/Static/audioset_20k_cleaned_weight.csv'
            samples_weight = np.loadtxt(weight_path_else, delimiter=',')
        elif args.dataset == 'kinetics':
            weight_path_else = '/home/amukherjee/Static/kin_tr_weight.csv'
            samples_weight = np.loadtxt(weight_path_else, delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    if args.dataset == 'vggsound':
        train_loader = torch.utils.data.DataLoader(
            dataloader.VGGSound(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    elif args.dataset == 'audioset':
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    elif args.dataset == 'kinetics':
        train_loader = torch.utils.data.DataLoader(
            dataloader.KineticsDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

else:
    print('balanced sampler is not used')
    if args.dataset == 'vggsound':
        train_loader = torch.utils.data.DataLoader(
        dataloader.VGGSound(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, four_tiles=args.four_tiling, restrict_randomness=args.restrict_randomness),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    elif args.dataset == 'audioset':
        train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, four_tiles=args.four_tiling, restrict_randomness=args.restrict_randomness),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    elif args.dataset == 'kinetics':
        train_loader = torch.utils.data.DataLoader(
        dataloader.KineticsDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, four_tiles=args.four_tiling, restrict_randomness=args.restrict_randomness),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.dataset == 'vggsound':
    val_loader = torch.utils.data.DataLoader(
        dataloader.VGGSound(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, four_tiles=args.four_tiling, restrict_randomness=args.restrict_randomness),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.data_eval != None:
        eval_loader = torch.utils.data.DataLoader(
            dataloader.VGGSound(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf, four_tiles=args.four_tiling, restrict_randomness=args.restrict_randomness),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
elif args.dataset == 'audioset':
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.data_eval != None:
        eval_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
elif args.dataset == 'kinetics':
    val_loader = torch.utils.data.DataLoader(
        dataloader.KineticsDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.data_eval != None:
        eval_loader = torch.utils.data.DataLoader(
            dataloader.KineticsDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.model == 'cav-mae':
    print('pretrain a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
    audio_model = models.CAVMAE(
    audio_length=args.target_length, 
    norm_pix_loss=args.norm_pix_loss, 
    modality_specific_depth=11, 
    tr_pos=args.tr_pos, 
    knowledge_distillation=args.knowledge_distillation, 
    k_value=args.k_value,
    dynamic_weighting=args.dynamic_weighting,
    model_type=args.model_type,
    dynamic_weight_normalization_method=args.dynamic_weight_normalization_method,
    absolute_noise=args.absolute_noise,
    split_decoder=args.split_decoder,
    dual_mask = args.dual_mask,
    complementary = args.complementary,
    AAVV = args.AAVV,
    triple_mask = args.triple_mask,
    kd_weight = args.kd_weight
    )
else:
    raise ValueError('model not supported')

# initialized with a pretrained checkpoint (e.g., original vision-MAE checkpoint)
if args.pretrain_path != 'None':
    mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print('now load mae pretrained weights from ', args.pretrain_path)
    print(miss, unexpected)

# if args.cont_model != None:
#     print('now load pretrained weights from : ' + args.cont_model)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sdA = torch.load(args.cont_model, map_location=device)
#     if isinstance(audio_model, torch.nn.DataParallel) == False:
#         audio_model = torch.nn.DataParallel(audio_model)
#     audio_model.load_state_dict(sdA, strict=True)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)
