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
from fvcore.nn import FlopCountAnalysis

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter, loss_audio_class_meter, loss_video_class_meter, loss_latent_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_loss,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # Create dummy tensors for a_input and v_input based on a real batch from train_loader
    # dummy_a_input, dummy_v_input, _ = next(iter(train_loader))
    # print(dummy_a_input.shape)
    # print(dummy_v_input.shape)
    # dummy_a_input = dummy_a_input.to(device)
    # dummy_v_input = dummy_v_input.to(device)

    """
    # Dummy label inputs
    labels = torch.rand(2, 527).to(device)

    # FLOP counting
    flops = FlopCountAnalysis(audio_model, (dummy_a_input, dummy_v_input, labels))
    total_flops = flops.total()
    print(f"Total FLOPs for the model: {total_flops}")
    """
    """
    # For Audio
    audio_encoder_parameters = [
        *list(audio_model.module.patch_embed_a.parameters()),
        audio_model.module.modality_a,
        audio_model.module.pos_embed_a,
        *[p for blk in audio_model.module.blocks_a for p in blk.parameters()]
    ]

    audio_decoder_parameters = [
        audio_model.module.decoder_modality_a,
        audio_model.module.decoder_pos_embed_a,
        *list(audio_model.module.decoder_pred_a.parameters())
    ]

    # For Video
    video_encoder_parameters = [
        *list(audio_model.module.patch_embed_v.parameters()),
        audio_model.module.modality_v,
        audio_model.module.pos_embed_v,
        *[p for blk in audio_model.module.blocks_v for p in blk.parameters()]
    ]

    video_decoder_parameters = [
        audio_model.module.decoder_modality_v,
        audio_model.module.decoder_pos_embed_v,
        *list(audio_model.module.decoder_pred_v.parameters())
    ]

    weighting_parameters = [
    audio_model.module.log_var_c,
    audio_model.module.log_var_latent,
    audio_model.module.log_var_v,
    audio_model.module.log_var_a
    ]
    

    # Combine for full streams
    audio_parameters = audio_encoder_parameters + audio_decoder_parameters
    video_parameters = video_encoder_parameters + video_decoder_parameters

    # For Joint parameters
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    
    all_parameters = list(audio_model.parameters())
    leaf_parameters = [p for p in all_parameters if p.is_leaf]
    non_leaf_parameters = [p for p in all_parameters if not p.is_leaf]
    print("Leaf parameters:")
    for param in leaf_parameters:
        print(param.shape)
    print("\nNon-Leaf parameters:")
    for param in non_leaf_parameters:
        print(param.shape)

    audio_ids = [id(p) for p in audio_parameters]
    video_ids = [id(p) for p in video_parameters]
    weighting_ids = [id(p) for p in weighting_parameters]
    joint_parameters = [p for p in trainables if id(p) not in audio_ids and id(p) not in video_ids and id(p) not in weighting_ids]
    """
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    """
    #encoder_optimizer = torch.optim.Adam(encoder_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    #decoder_optimizer = torch.optim.Adam(decoder_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    audio_optimizer = torch.optim.Adam(audio_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    video_optimizer = torch.optim.Adam(video_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    joint_optimizer = torch.optim.Adam(joint_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    weighting_optimizer = torch.optim.Adam(weighting_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    """
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

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    audio_model.train()
    # Initialization outside the loop
    weights = torch.tensor([1/5]*5).cuda() # Assuming you have 5 loss components and you're using CUDA
    prev_losses = torch.zeros(5).cuda()
    torch.manual_seed(42)  # You can use any number as the seed
    prev_weights = torch.rand(5).cuda()   # Create a tensor of random values between 0 and 1
    prev_weights /= prev_weights.sum()    # Normalize to ensure the sum is 1
    weights.requires_grad = False
    prev_losses.requires_grad = False
    prev_weights.requires_grad = False

    lr = 1e-3

    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('current masking ratio is {:.3f} for both modalities; audio mask mode {:s}'.format(args.masking_ratio, args.mask_mode))

        for i, (a_input, v_input, label_indices) in enumerate(train_loader):

            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            label_indices = label_indices.to(device, non_blocking=True)
            
            # Convert one-hot encoded labels to class indices
            audio_labels = torch.argmax(label_indices, dim=1)
            video_labels = audio_labels
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()
            if audio_labels is None:
                raise ValueError(f"labels is None!")
            print(f"audio_labels: {audio_labels}, Shape: {audio_labels.shape}")
            
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_latent,audio_logits, video_logits, audio_classification_loss, video_classification_loss= audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio,labels=audio_labels, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_latent, audio_classification_loss, video_classification_loss = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean(), loss_latent.sum(), audio_classification_loss.sum(), video_classification_loss.sum()

            optimizer.zero_grad()

            all_losses = torch.stack([loss_mae, loss_c, loss_latent, audio_classification_loss, video_classification_loss])
            del_losses = all_losses - prev_losses
            del_weights = weights - prev_weights
            
            delta_weights = del_losses / (del_weights + 1e-8)
            weights = weights - lr * delta_weights
            weights = weights / weights.sum()
            # Update the history of losses and weights for the next epoch
            prev_losses = all_losses.clone()
            prev_weights = weights.clone()

            total_loss = torch.dot(weights.detach(), all_losses)
            scaler.scale(total_loss).backward() # Use the computed total_loss for backpropagation

            scaler.step(optimizer)
            scaler.update()

            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            loss_audio_class_meter.update(audio_classification_loss.item(), B)
            loss_video_class_meter.update(video_classification_loss.item(), B)
            loss_latent_meter.update(loss_latent.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

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

        # print(f"weights: {weights}")
        # print(f"losses: {all_losses}")
        # print(f"delta_weights: {delta_weights}")
        # print(f"del_losses: {del_losses}")
        # print(f"del_weights: {del_weights}")
        # print(f"prev_losses: {prev_losses}")
        # print(f"prev_weights: {prev_weights}")

        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc, audio_classification_loss, video_classification_loss = validate(audio_model, test_loader, args)

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
        print("Train Latent Loss: {:.6f}".format(loss_latent_meter.avg))
        print("Train Audio Classification Loss: {:.6f}".format(loss_audio_class_meter.avg))
        print("Train Video Classification Loss: {:.6f}".format(loss_video_class_meter.avg))

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

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
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()

def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, label_indices) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            label_indices = label_indices.to(device, non_blocking=True)
            
            # Convert one-hot encoded labels to class indices
            audio_labels = torch.argmax(label_indices, dim=1)
            video_labels = audio_labels

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_latent,audio_logits, video_logits, audio_classification_loss, video_classification_loss= audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio,labels=audio_labels, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_latent, audio_classification_loss, video_classification_loss = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean(), loss_latent.sum(), audio_classification_loss.sum(), video_classification_loss.sum()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, audio_classification_loss, video_classification_loss
