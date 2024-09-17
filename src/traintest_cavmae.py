# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# not rely on supervised feature

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter1, loss_a_meter1, loss_v_meter1, loss_c_meter1 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter2, loss_a_meter2, loss_v_meter2, loss_c_meter2 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter3, loss_a_meter3, loss_v_meter3, loss_c_meter3 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_kd_meter = AverageMeter()

    progress = []

    best_epoch_acc,best_epoch_loss, best_loss = 0, 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch_loss, best_loss,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    def _load_progress():
        with open("%s/progress.pkl" % exp_dir, "rb") as f:
            progress_to_return = pickle.load(f)
        return progress_to_return

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    # FLop Count
    dummy_a_input, dummy_v_input, *_ = next(iter(train_loader))
    print(dummy_a_input.shape)
    print(dummy_v_input.shape)
    dummy_a_input = dummy_a_input.to(device)
    dummy_v_input = dummy_v_input.to(device)
    audio_model = audio_model.to(device)
    flops = FlopCountAnalysis(audio_model, (1, dummy_a_input, dummy_v_input))
    total_flops = flops.total()
    print(f"Total FLOPs for the model: {total_flops}") 

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

    epoch += 1
    scaler = GradScaler()

    if args.resume_from_checkpoint != "":
        print("RESUMING CKPT.....")
        checkpoint = torch.load(args.resume_from_checkpoint)
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Resuming from checkpoint at epoch {epoch}...")
        progress = _load_progress()

    audio_model = audio_model.to(device)

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    writer = SummaryWriter(log_dir=f"{args.exp_dir}/runs")
    if args.dual_mask == True:
        result = np.zeros([args.n_epochs, 13])  # for each epoch, 13 metrics to record
    elif args.triple_mask == True:
        result = np.zeros([args.n_epochs, 17])
    else:
        result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    print("Done ")
    exit(1) # For Flop count only 
    audio_model.train()
    while epoch < args.n_epochs + 1:
        train_loader.dataset.next_epoch()
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('current masking ratio is {:.3f} for both modalities; audio mask mode {:s}'.format(args.masking_ratio, args.mask_mode))

        for i, (a_input, v_input, label_indices, *masks) in enumerate(train_loader):
            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            if args.restrict_randomness:
                audio_mask, video_mask = masks
            else:
                audio_mask, video_mask = None, None

            if args.dual_mask == True:
                with autocast():
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, mask_a_1, mask_v_1, c_acc_1, loss_mae_2,loss_mae_a_2, loss_mae_v_2, loss_c_2, mask_a_2, mask_v_2, c_acc_2, loss_kd = audio_model(epoch-1, a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode, video_mask=video_mask, audio_mask=audio_mask)
                    # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, c_acc_1, loss_mae_2, loss_mae_a_2, loss_mae_v_2, loss_c_2, c_acc_2, loss_kd = loss.sum(), loss_mae_1.sum(), loss_mae_a_1.sum(), loss_mae_v_1.sum(), loss_c_1.sum(), c_acc_1.mean(), loss_mae_2.sum(), loss_mae_a_2.sum(), loss_mae_v_2.sum(), loss_c_2.sum(), c_acc_2.mean(), loss_kd.sum()

            elif args.triple_mask == True:
                with autocast():
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, mask_a_1, mask_v_1, c_acc_1, loss_mae_2,loss_mae_a_2, loss_mae_v_2, loss_c_2, mask_a_2, mask_v_2, c_acc_2, loss_mae_3,loss_mae_a_3, loss_mae_v_3, loss_c_3, mask_a_3, mask_v_3, c_acc_3, loss_kd = audio_model(epoch-1, a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode, video_mask=video_mask, audio_mask=audio_mask)
                    # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, c_acc_1, loss_mae_2, loss_mae_a_2, loss_mae_v_2, loss_c_2, c_acc_2, loss_mae_3, loss_mae_a_3, loss_mae_v_3, loss_c_3, c_acc_3, loss_kd = loss.sum(), loss_mae_1.sum(), loss_mae_a_1.sum(), loss_mae_v_1.sum(), loss_c_1.sum(), c_acc_1.mean(), loss_mae_2.sum(), loss_mae_a_2.sum(), loss_mae_v_2.sum(), loss_c_2.sum(), c_acc_2.mean(), loss_mae_3.sum(), loss_mae_a_3.sum(), loss_mae_v_3.sum(), loss_c_3.sum(), c_acc_3.mean(), loss_kd.sum()

            else:
                with autocast():
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(epoch-1, a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode, video_mask=video_mask, audio_mask=audio_mask)
                    # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_av is the main loss
            if args.dual_mask == True:
                loss_av_meter1.update(loss.item(), B)
                loss_a_meter1.update(loss_mae_1.item(), B)
                loss_v_meter1.update(loss_mae_v_1.item(), B)
                loss_c_meter1.update(loss_c_1.item(), B)
                loss_av_meter2.update(loss.item(), B)
                loss_a_meter2.update(loss_mae_2.item(), B)
                loss_v_meter2.update(loss_mae_v_2.item(), B)
                loss_c_meter2.update(loss_c_2.item(), B)
                loss_kd_meter.update(loss_kd.item(), B)
            elif args.triple_mask == True:
                loss_av_meter1.update(loss.item(), B)
                loss_a_meter1.update(loss_mae_1.item(), B)
                loss_v_meter1.update(loss_mae_v_1.item(), B)
                loss_c_meter1.update(loss_c_1.item(), B)
                loss_av_meter2.update(loss.item(), B)
                loss_a_meter2.update(loss_mae_2.item(), B)
                loss_v_meter2.update(loss_mae_v_2.item(), B)
                loss_c_meter2.update(loss_c_2.item(), B)
                loss_av_meter3.update(loss.item(), B)
                loss_a_meter3.update(loss_mae_3.item(), B)
                loss_v_meter3.update(loss_mae_v_3.item(), B)
                loss_c_meter3.update(loss_c_3.item(), B)
                loss_kd_meter.update(loss_kd.item(), B)
            else:
                loss_av_meter.update(loss.item(), B)
                loss_a_meter.update(loss_mae_a.item(), B)
                loss_v_meter.update(loss_mae_v.item(), B)
                loss_c_meter.update(loss_c.item(), B)

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if args.dual_mask == True:
                if print_step and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                    'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                    'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                    'Train Total Loss 1 {loss_av_meter1.val:.4f}\t'
                    'Train MAE Loss Audio 1 {loss_a_meter1.val:.4f}\t'
                    'Train MAE Loss Visual 1 {loss_v_meter1.val:.4f}\t'
                    'Train Contrastive Loss 1 {loss_c_meter1.val:.4f}\t'
                    'Train Total Loss 2 {loss_av_meter2.val:.4f}\t'
                    'Train MAE Loss Audio 2 {loss_a_meter2.val:.4f}\t'
                    'Train MAE Loss Visual 2 {loss_v_meter2.val:.4f}\t'
                    'Train Contrastive Loss 2 {loss_c_meter2.val:.4f}\t'
                    'Train KD Loss {loss_kd_meter.val:.4f}\t'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                        per_sample_dnn_time=per_sample_dnn_time, loss_av_meter1=loss_av_meter1, loss_a_meter1=loss_a_meter1, loss_v_meter1=loss_v_meter1, loss_c_meter1=loss_c_meter1, loss_av_meter2=loss_av_meter2, loss_a_meter2=loss_a_meter2, loss_v_meter2=loss_v_meter2, loss_c_meter2=loss_c_meter2, loss_kd_meter=loss_kd_meter), flush=True)
                    if np.isnan(loss_av_meter1.avg) or np.isnan(loss_av_meter2.avg):
                        print("training diverged...")
                        return
                    
                    # print("Parameter A", audio_model.module.log_noise_a)
                    # print("Parameter V", audio_model.module.log_noise_v)
                    # print("Parameter C", audio_model.module.log_noise_c)
                    # print("Parameter AA", audio_model.module.log_noise_AA)
                    # print("Parameter VV", audio_model.module.log_noise_VV)
                    # print("Parameter AV", audio_model.module.log_noise_AV)

                    # Step 1: Stack the parameters
                    # weights = torch.stack([
                    #     # audio_model.module.log_noise_a, 
                    #     # audio_model.module.log_noise_v, 
                    #     audio_model.module.log_noise_c, 
                    #     audio_model.module.log_noise_AA, 
                    #     audio_model.module.log_noise_VV, 
                    #     audio_model.module.log_noise_AV
                    # ])
                    
                    # # Step 2: Apply softmax
                    # softmax_weights = F.softmax(weights, dim=0) * 4
                    
                    # # Step 3: Print the softmax-normalized weights
                    # # print("Softmax Weight mae_a:", softmax_weights[0].item())
                    # # print("Softmax Weight mae_v:", softmax_weights[1].item())
                    # print("Softmax Weight contrastive:", softmax_weights[0].item())
                    # print("Softmax Weight AA:", softmax_weights[1].item())
                    # print("Softmax Weight VV:", softmax_weights[2].item())
                    # print("Softmax Weight AV:", softmax_weights[3].item())

            elif args.triple_mask == True:
                if print_step and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                    'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                    'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                    'Train Total Loss 1 {loss_av_meter1.val:.4f}\t'
                    'Train MAE Loss Audio 1 {loss_a_meter1.val:.4f}\t'
                    'Train MAE Loss Visual 1 {loss_v_meter1.val:.4f}\t'
                    'Train Contrastive Loss 1 {loss_c_meter1.val:.4f}\t'
                    'Train Total Loss 2 {loss_av_meter2.val:.4f}\t'
                    'Train MAE Loss Audio 2 {loss_a_meter2.val:.4f}\t'
                    'Train MAE Loss Visual 2 {loss_v_meter2.val:.4f}\t'
                    'Train Contrastive Loss 2 {loss_c_meter2.val:.4f}\t'
                    'Train Total Loss 3 {loss_av_meter3.val:.4f}\t'
                    'Train MAE Loss Audio 3 {loss_a_meter3.val:.4f}\t'
                    'Train MAE Loss Visual 3 {loss_v_meter3.val:.4f}\t'
                    'Train Contrastive Loss 3 {loss_c_meter3.val:.4f}\t'
                    'Train KD Loss {loss_kd_meter.val:.4f}\t'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                        per_sample_dnn_time=per_sample_dnn_time, loss_av_meter1=loss_av_meter1, loss_a_meter1=loss_a_meter1, loss_v_meter1=loss_v_meter1, loss_c_meter1=loss_c_meter1, loss_av_meter2=loss_av_meter2, loss_a_meter2=loss_a_meter2, loss_v_meter2=loss_v_meter2, loss_c_meter2=loss_c_meter2, loss_av_meter3=loss_av_meter3, loss_a_meter3=loss_a_meter3, loss_v_meter3=loss_v_meter3, loss_c_meter3 = loss_c_meter3, loss_kd_meter=loss_kd_meter), flush=True)
            else:
                if print_step and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                    'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                    'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                    'Train Total Loss {loss_av_meter.val:.4f}\t'
                    'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
                    'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                    'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                    'Train Contrastive Acc {c_acc:.3f}\t'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                        per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                    if np.isnan(loss_av_meter.avg):
                        print("training diverged...")
                        return

            end_time = time.time()
            global_step += 1

        print('start validation')
        if args.dual_mask == True:
            eval_loss_av, eval_loss_mae1, eval_loss_mae_a1, eval_loss_mae_v1, eval_loss_c1, eval_c_acc1, eval_loss_mae2, eval_loss_mae_a2, eval_loss_mae_v2, eval_loss_c2, eval_c_acc2, eval_loss_kd = validate(audio_model, test_loader, args)

            print("Eval Audio MAE Loss 1: {:.6f}".format(eval_loss_mae_a1))
            print("Eval Visual MAE Loss 1: {:.6f}".format(eval_loss_mae_v1))
            print("Eval Total MAE Loss 1: {:.6f}".format(eval_loss_mae1))
            print("Eval Contrastive Loss 1: {:.6f}".format(eval_loss_c1))
            print("Eval Contrastive Accuracy 1: {:.6f}".format(eval_c_acc1))
            print("Eval Audio MAE Loss 2: {:.6f}".format(eval_loss_mae_a2))
            print("Eval Visual MAE Loss 2: {:.6f}".format(eval_loss_mae_v2))
            print("Eval Total MAE Loss 2: {:.6f}".format(eval_loss_mae2))
            print("Eval Contrastive Loss 2: {:.6f}".format(eval_loss_c2))
            print("Eval Contrastive Accuracy 2: {:.6f}".format(eval_c_acc2))
            print("Eval KD Loss: {:.6f}".format(eval_loss_kd))
            print("Eval Total Loss : {:.6f}".format(eval_loss_av))

            print("Train Audio MAE Loss 1: {:.6f}".format(loss_a_meter1.avg))
            print("Train Visual MAE Loss 1: {:.6f}".format(loss_v_meter1.avg))
            print("Train Contrastive Loss 1: {:.6f}".format(loss_c_meter1.avg))
            print("Train Total Loss 1: {:.6f}".format(loss_av_meter1.avg))
            print("Train Audio MAE Loss 2: {:.6f}".format(loss_a_meter2.avg))
            print("Train Visual MAE Loss 2: {:.6f}".format(loss_v_meter2.avg))
            print("Train Contrastive Loss 2: {:.6f}".format(loss_c_meter2.avg))
            print("Train KD Loss: {:.6f}".format(loss_kd_meter.avg))
            print("Train Total Loss 2: {:.6f}".format(loss_av_meter2.avg))

            writer.add_scalar('Validation Loss/Total Loss', eval_loss_av, epoch)
            writer.add_scalar('Validation Loss/Audio MAE Loss 1', eval_loss_mae_a1, epoch)
            writer.add_scalar('Validation Loss/Visual MAE Loss 1', eval_loss_mae_v1, epoch)
            writer.add_scalar('Validation Loss/Total MAE Loss 1', eval_loss_mae1, epoch)
            writer.add_scalar('Validation Loss/Contrastive Loss 1', eval_loss_c1, epoch)
            writer.add_scalar('Validation Accuracy/Contrastive Accuracy 1', eval_c_acc1, epoch)
            writer.add_scalar('Validation Loss/Total Loss 2', eval_loss_av, epoch)
            writer.add_scalar('Validation Loss/Audio MAE Loss 2', eval_loss_mae_a2, epoch)
            writer.add_scalar('Validation Loss/Visual MAE Loss 2', eval_loss_mae_v2, epoch)
            writer.add_scalar('Validation Loss/Total MAE Loss 2', eval_loss_mae2, epoch)
            writer.add_scalar('Validation Loss/Contrastive Loss 2', eval_loss_c2, epoch)
            writer.add_scalar('Validation Accuracy/Contrastive Accuracy 2', eval_c_acc2, epoch)
            writer.add_scalar('Validation Loss/KD Loss', eval_loss_kd, epoch)

            # print("Epoch:", epoch, "Noise Parameters:", audio_model.multi_noise_loss_module.noise_params)
        elif args.triple_mask == True:
            eval_loss_av, eval_loss_mae1, eval_loss_mae_a1, eval_loss_mae_v1, eval_loss_c1, eval_c_acc1, eval_loss_mae2, eval_loss_mae_a2, eval_loss_mae_v2, eval_loss_c2, eval_c_acc2, eval_loss_mae3, eval_loss_mae_a3, eval_loss_mae_v3, eval_loss_c3, eval_c_acc3, eval_loss_kd = validate(audio_model, test_loader, args)

            print("Eval Audio MAE Loss 1: {:.6f}".format(eval_loss_mae_a1))
            print("Eval Visual MAE Loss 1: {:.6f}".format(eval_loss_mae_v1))
            print("Eval Total MAE Loss 1: {:.6f}".format(eval_loss_mae1))
            print("Eval Contrastive Loss 1: {:.6f}".format(eval_loss_c1))
            print("Eval Contrastive Accuracy 1: {:.6f}".format(eval_c_acc1))
            print("Eval Audio MAE Loss 2: {:.6f}".format(eval_loss_mae_a2))
            print("Eval Visual MAE Loss 2: {:.6f}".format(eval_loss_mae_v2))
            print("Eval Total MAE Loss 2: {:.6f}".format(eval_loss_mae2))
            print("Eval Contrastive Loss 2: {:.6f}".format(eval_loss_c2))
            print("Eval Contrastive Accuracy 2: {:.6f}".format(eval_c_acc2))
            print("Eval Audio MAE Loss 3: {:.6f}".format(eval_loss_mae_a3))
            print("Eval Visual MAE Loss 3: {:.6f}".format(eval_loss_mae_v3))
            print("Eval Total MAE Loss 3: {:.6f}".format(eval_loss_mae3))
            print("Eval Contrastive Loss 3: {:.6f}".format(eval_loss_c3))
            print("Eval Contrastive Accuracy 3: {:.6f}".format(eval_c_acc3))
            
            print("Eval KD Loss: {:.6f}".format(eval_loss_kd))
            print("Eval Total Loss : {:.6f}".format(eval_loss_av))
            print("Train Audio MAE Loss 1: {:.6f}".format(loss_a_meter1.avg))
            print("Train Visual MAE Loss 1: {:.6f}".format(loss_v_meter1.avg))
            print("Train Contrastive Loss 1: {:.6f}".format(loss_c_meter1.avg))
            print("Train Total Loss 1: {:.6f}".format(loss_av_meter1.avg))
            print("Train Audio MAE Loss 2: {:.6f}".format(loss_a_meter2.avg))
            print("Train Visual MAE Loss 2: {:.6f}".format(loss_v_meter2.avg))
            print("Train Contrastive Loss 2: {:.6f}".format(loss_c_meter2.avg))
            print("Train KD Loss: {:.6f}".format(loss_kd_meter.avg))
            print("Train Total Loss 2: {:.6f}".format(loss_av_meter2.avg))

            writer.add_scalar('Validation Loss/Total Loss', eval_loss_av, epoch)
            writer.add_scalar('Validation Loss/Audio MAE Loss 1', eval_loss_mae_a1, epoch)
            writer.add_scalar('Validation Loss/Visual MAE Loss 1', eval_loss_mae_v1, epoch)
            writer.add_scalar('Validation Loss/Total MAE Loss 1', eval_loss_mae1, epoch)
            writer.add_scalar('Validation Loss/Contrastive Loss 1', eval_loss_c1, epoch)
            writer.add_scalar('Validation Accuracy/Contrastive Accuracy 1', eval_c_acc1, epoch)
            writer.add_scalar('Validation Loss/Total Loss 2', eval_loss_av, epoch)
            writer.add_scalar('Validation Loss/Audio MAE Loss 2', eval_loss_mae_a2, epoch)
            writer.add_scalar('Validation Loss/Visual MAE Loss 2', eval_loss_mae_v2, epoch)
            writer.add_scalar('Validation Loss/Total MAE Loss 2', eval_loss_mae2, epoch)
            writer.add_scalar('Validation Loss/Contrastive Loss 2', eval_loss_c2, epoch)
            writer.add_scalar('Validation Accuracy/Contrastive Accuracy 2', eval_c_acc2, epoch)
            writer.add_scalar('Validation Loss/KD Loss', eval_loss_kd, epoch)
            writer.add_scalar('Validation Loss/Audio MAE Loss 3', eval_loss_mae_a3, epoch)
            writer.add_scalar('Validation Loss/Visual MAE Loss 3', eval_loss_mae_v3, epoch)
            writer.add_scalar('Validation Loss/Total MAE Loss 3', eval_loss_mae3, epoch)
            writer.add_scalar('Validation Loss/Contrastive Loss 3', eval_loss_c3, epoch)
            writer.add_scalar('Validation Accuracy/Contrastive Accuracy 3', eval_c_acc3, epoch)

        else:
            eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(audio_model, test_loader, args)

            print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
            print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
            print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
            print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
            print("Eval Total Loss: {:.6f}".format(eval_loss_av))
            print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

            print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
            print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
            print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
            print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))

            writer.add_scalar('Validation Loss/Total Loss', eval_loss_av, epoch)
            writer.add_scalar('Validation Loss/Audio MAE Loss', eval_loss_mae_a, epoch)
            writer.add_scalar('Validation Loss/Visual MAE Loss', eval_loss_mae_v, epoch)
            writer.add_scalar('Validation Loss/Total MAE Loss', eval_loss_mae, epoch)
            writer.add_scalar('Validation Loss/Contrastive Loss', eval_loss_c, epoch)
            writer.add_scalar('Validation Accuracy/Contrastive Accuracy', eval_c_acc, epoch)
            
            writer.add_scalar('Train Loss/Total Loss', loss_av_meter.avg, epoch)
            writer.add_scalar('Train Loss/Audio MAE Loss', loss_a_meter.avg, epoch)
            writer.add_scalar('Train Loss/Visual MAE Loss', loss_v_meter.avg, epoch)
            writer.add_scalar('Train Loss/Contrastive Loss', loss_c_meter.avg, epoch)
            writer.add_scalar('Train Accuracy/Contrastive Accuracy', c_acc, epoch)

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        # if args.dual_mask == True:
        #     result[epoch-1, :] = [loss_a_meter1.avg, loss_v_meter1.avg, loss_c_meter1.avg, loss_av_meter1.avg, loss_a_meter2.avg, loss_v_meter2.avg, loss_c_meter2.avg, loss_av_meter2.avg, eval_loss_mae_a1, eval_loss_mae_v1, eval_loss_c1, eval_loss_av, eval_c_acc1, optimizer.param_groups[0]['lr']]
        # else:
        #     result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        # np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch_loss = epoch

        if best_epoch_loss == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        # to be sure leaving both uncommented
        if best_epoch_loss == epoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': audio_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # Include any other metrics or states you care about
                'loss': best_epoch_loss,
            }, "%s/models/best_checkpoint.pth" % exp_dir)
            # save the scheduler
            torch.save(scheduler.state_dict(), "%s/models/best_scheduler_state.pth" % (exp_dir))
        # if eval_c_acc > best_epoch_acc:
        #     best_epoch_acc = eval_c_acc
        #     torch.save(audio_model.state_dict(), "%s/models/best_audio_model_acc.pth" % (exp_dir))
        #     torch.save(optimizer.state_dict(), "%s/models/best_optim_state_acc.pth" % (exp_dir))
        
        # save model every epoch
        if epoch % 1 == 0:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': audio_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # You can also include additional information here if necessary
            }, "%s/models/checkpoint.%d.pth" % (exp_dir, epoch))
        # if args.save_model == True:
        #     torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()

        if args.dual_mask == True:
            loss_av_meter1.reset()
            loss_a_meter1.reset()
            loss_v_meter1.reset()
            loss_c_meter1.reset()
            loss_av_meter2.reset()
            loss_a_meter2.reset()
            loss_v_meter2.reset()
            loss_c_meter2.reset()
        elif args.triple_mask == True:
            loss_av_meter1.reset()
            loss_a_meter1.reset
            loss_v_meter1.reset()
            loss_c_meter1.reset()
            loss_av_meter2.reset()
            loss_a_meter2.reset()
            loss_v_meter2.reset()
            loss_c_meter2.reset()
            loss_av_meter3.reset()
            loss_a_meter3.reset()
            loss_v_meter3.reset()
            loss_c_meter3.reset()
        else:
            loss_av_meter.reset()
            loss_a_meter.reset()
            loss_v_meter.reset()
            loss_c_meter.reset()
    
    writer.close()

def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    if args.dual_mask == True:
        A_loss, A_loss_mae1, A_loss_mae_a1, A_loss_mae_v1, A_loss_c1, A_c_acc1,A_loss_mae2, A_loss_mae_a2, A_loss_mae_v2, A_loss_c2, A_c_acc2, A_loss_kd  = [], [], [], [], [], [], [], [], [], [], [], []
    elif args.triple_mask == True:
        A_loss, A_loss_mae1, A_loss_mae_a1, A_loss_mae_v1, A_loss_c1, A_c_acc1,A_loss_mae2, A_loss_mae_a2, A_loss_mae_v2, A_loss_c2, A_c_acc2, A_loss_mae3, A_loss_mae_a3, A_loss_mae_v3, A_loss_c3, A_c_acc3, A_loss_kd  = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    else:
        A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, label_indices, *masks) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            if args.restrict_randomness == True:
                audio_mask, video_mask = masks
            else:
                audio_mask, video_mask = None, None

            if args.dual_mask == True:
                with autocast():
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, mask_a_1, mask_v_1, c_acc_1, loss_mae_2,loss_mae_a_2, loss_mae_v_2, loss_c_2, mask_a_2, mask_v_2, c_acc_2, loss_kd  = audio_model(-1, a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode, audio_mask=audio_mask, video_mask=video_mask)
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, c_acc_1, loss_mae_2, loss_mae_a_2, loss_mae_v_2, loss_c_2, c_acc_2, loss_kd = loss.sum(), loss_mae_1.sum(), loss_mae_a_1.sum(), loss_mae_v_1.sum(), loss_c_1.sum(), c_acc_1.mean(), loss_mae_2.sum(), loss_mae_a_2.sum(), loss_mae_v_2.sum(), loss_c_2.sum(), c_acc_2.mean(), loss_kd.sum()
            elif args.triple_mask == True:
                with autocast():
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, mask_a_1, mask_v_1, c_acc_1, loss_mae_2,loss_mae_a_2, loss_mae_v_2, loss_c_2, mask_a_2, mask_v_2, c_acc_2, loss_mae_3,loss_mae_a_3, loss_mae_v_3, loss_c_3, mask_a_3, mask_v_3, c_acc_3, loss_kd = audio_model(-1, a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode, video_mask=video_mask, audio_mask=audio_mask)
                    # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                    loss, loss_mae_1, loss_mae_a_1, loss_mae_v_1, loss_c_1, c_acc_1, loss_mae_2, loss_mae_a_2, loss_mae_v_2, loss_c_2, c_acc_2, loss_mae_3, loss_mae_a_3, loss_mae_v_3, loss_c_3, c_acc_3, loss_kd = loss.sum(), loss_mae_1.sum(), loss_mae_a_1.sum(), loss_mae_v_1.sum(), loss_c_1.sum(), c_acc_1.mean(), loss_mae_2.sum(), loss_mae_a_2.sum(), loss_mae_v_2.sum(), loss_c_2.sum(), c_acc_2.mean(), loss_mae_3.sum(), loss_mae_a_3.sum(), loss_mae_v_3.sum(), loss_c_3.sum(), c_acc_3.mean(), loss_kd.sum()
            else:
                with autocast():
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(-1, a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode, audio_mask=audio_mask, video_mask=video_mask)
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            
            if args.dual_mask == True:
                A_loss.append(loss.to('cpu').detach())
                A_loss_mae1.append(loss_mae_1.to('cpu').detach())
                A_loss_mae_a1.append(loss_mae_a_1.to('cpu').detach())
                A_loss_mae_v1.append(loss_mae_v_1.to('cpu').detach())
                A_loss_c1.append(loss_c_1.to('cpu').detach())
                A_c_acc1.append(c_acc_1.to('cpu').detach())
                A_loss_mae2.append(loss_mae_2.to('cpu').detach())
                A_loss_mae_a2.append(loss_mae_a_2.to('cpu').detach())
                A_loss_mae_v2.append(loss_mae_v_2.to('cpu').detach())
                A_loss_c2.append(loss_c_2.to('cpu').detach())
                A_c_acc2.append(c_acc_2.to('cpu').detach())
                A_loss_kd.append(loss_kd.to('cpu').detach())
                batch_time.update(time.time() - end)
                end = time.time()
            elif args.triple_mask == True:
                A_loss.append(loss.to('cpu').detach())
                A_loss_mae1.append(loss_mae_1.to('cpu').detach())
                A_loss_mae_a1.append(loss_mae_a_1.to('cpu').detach())
                A_loss_mae_v1.append(loss_mae_v_1.to('cpu').detach())
                A_loss_c1.append(loss_c_1.to('cpu').detach())
                A_c_acc1.append(c_acc_1.to('cpu').detach())
                A_loss_mae2.append(loss_mae_2.to('cpu').detach())
                A_loss_mae_a2.append(loss_mae_a_2.to('cpu').detach())
                A_loss_mae_v2.append(loss_mae_v_2.to('cpu').detach())
                A_loss_c2.append(loss_c_2.to('cpu').detach())
                A_c_acc2.append(c_acc_2.to('cpu').detach())
                A_loss_mae3.append(loss_mae_3.to('cpu').detach())
                A_loss_mae_a3.append(loss_mae_a_3.to('cpu').detach())
                A_loss_mae_v3.append(loss_mae_v_3.to('cpu').detach())
                A_loss_c3.append(loss_c_3.to('cpu').detach())
                A_c_acc3.append(c_acc_3.to('cpu').detach())
                A_loss_kd.append(loss_kd.to('cpu').detach())
                batch_time.update(time.time() - end)
                end = time.time()
            else:
                A_loss.append(loss.to('cpu').detach())
                A_loss_mae.append(loss_mae.to('cpu').detach())
                A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
                A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
                A_loss_c.append(loss_c.to('cpu').detach())
                A_c_acc.append(c_acc.to('cpu').detach())
                batch_time.update(time.time() - end)
                end = time.time()

        if args.dual_mask == True:
            loss = np.mean(A_loss)
            loss_mae1 = np.mean(A_loss_mae1)
            loss_mae_a1 = np.mean(A_loss_mae_a1)
            loss_mae_v1 = np.mean(A_loss_mae_v1)
            loss_c1 = np.mean(A_loss_c1)
            c_acc1 = np.mean(A_c_acc1)
            loss_mae2 = np.mean(A_loss_mae2)
            loss_mae_a2 = np.mean(A_loss_mae_a2)
            loss_mae_v2 = np.mean(A_loss_mae_v2)
            loss_c2 = np.mean(A_loss_c2)
            c_acc2 = np.mean(A_c_acc2)
            loss_kd = np.mean(A_loss_kd)
            return loss, loss_mae1, loss_mae_a1, loss_mae_v1, loss_c1, c_acc1, loss_mae2, loss_mae_a2, loss_mae_v2, loss_c2, c_acc2, loss_kd
        elif args.triple_mask == True:
            loss = np.mean(A_loss)
            loss_mae1 = np.mean(A_loss_mae1)
            loss_mae_a1 = np.mean(A_loss_mae_a1)
            loss_mae_v1 = np.mean(A_loss_mae_v1)
            loss_c1 = np.mean(A_loss_c1)
            c_acc1 = np.mean(A_c_acc1)
            loss_mae2 = np.mean(A_loss_mae2)
            loss_mae_a2 = np.mean(A_loss_mae_a2)
            loss_mae_v2 = np.mean(A_loss_mae_v2)
            loss_c2 = np.mean(A_loss_c2)
            c_acc2 = np.mean(A_c_acc2)
            loss_mae3 = np.mean(A_loss_mae3)
            loss_mae_a3 = np.mean(A_loss_mae_a3)
            loss_mae_v3 = np.mean(A_loss_mae_v3)
            loss_c3 = np.mean(A_loss_c3)
            c_acc3 = np.mean(A_c_acc3)
            loss_kd = np.mean(A_loss_kd)
            return loss, loss_mae1, loss_mae_a1, loss_mae_v1, loss_c1, c_acc1, loss_mae2, loss_mae_a2, loss_mae_v2, loss_c2, c_acc2, loss_mae3, loss_mae_a3, loss_mae_v3, loss_c3, c_acc3, loss_kd
        else:
            loss = np.mean(A_loss)
            loss_mae = np.mean(A_loss_mae)
            loss_mae_a = np.mean(A_loss_mae_a)
            loss_mae_v = np.mean(A_loss_mae_v)
            loss_c = np.mean(A_loss_c)
            c_acc = np.mean(A_c_acc)
            return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc