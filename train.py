import argparse
import logging
import os
import numpy as np

np.random.seed(0)
import json
import pylab
from bisect import bisect

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from torch import nn
from torch.utils.data import DataLoader

from utils.train_utils import WarmUpLR, initialize_xavier, progress_bar
from utils.dataset import SlicerDataset, SlicerDatasetSNR
from modules.loss import reg_BCELoss
from modules.resnet import ResNet54Double
from modules.resnet import CNN
from modules.dain import DAIN_Layer
from modules.whiten import CropWhitenNet


def decode_snr_schedule(sch_str):
    steps = sch_str.split(',')
    epochs = []
    s_ranges = []
    for step in steps:
        ep, range_ = step.split(':')
        epochs.append(int(ep))
        s_min, s_max = range_.split('-')
        s_ranges.append([float(s_min), float(s_max)])
    epochs.append(1)
    s_ranges.append(([-np.inf, np.inf]))
    epochs = np.cumsum(epochs)
    return epochs, s_ranges


def get_snr_by_epoch(sch_epochs, sch_ranges, epoch):
    """
    Get SNR range for a given epoch number.
    
    IMPORTANT: epoch should be the ACTUAL epoch number (0-indexed or 1-indexed consistently)
    """
    print(f'Epoch: {epoch}, schedule: {[(sch_epoch, sch_range) for sch_epoch, sch_range in zip(sch_epochs, sch_ranges)]}')
    index = bisect(sch_epochs, epoch)
    if index >= len(sch_epochs):
        return sch_ranges[-1]
    return sch_ranges[index]


# Set default weights filename
default_weights_fname = 'weights.pt'

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def get_model(model_name, device):
    """
    Factory function to create the requested model.
    
    Arguments
    ---------
    model_name : str
        Name of the model architecture ('cnn' or 'resnet')
    device : str
        Device to place the model on
        
    Returns
    -------
    model : nn.Module
        The requested model
    """
    if model_name.lower() == 'cnn':
        print(f'Using CNN architecture')
        return CNN(detectors=2).to(device)
    elif model_name.lower() == 'resnet':
        print(f'Using ResNet54Double architecture')
        return ResNet54Double().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'cnn' or 'resnet'")


def save_checkpoint(epoch, net, opt, sch, train_losses, val_losses, 
                   train_accs, val_accs, output_dir, args):
    """
    Save COMPLETE checkpoint including model, optimizer, and scheduler state.
    This allows perfect resumption of training.
    
    NOTE: This should be called BEFORE scheduler.step() so that when resumed,
    the scheduler is in the correct state for the next epoch.
    """
    checkpoint = {
        # Model weights
        'model_state_dict': net.state_dict(),
        
        # *** CRITICAL: Optimizer state (momentum, etc.) ***
        'optimizer_state_dict': opt.state_dict(),
        
        # *** CRITICAL: Scheduler state (current LR) ***
        'scheduler_state_dict': sch.state_dict(),
        
        # Training progress - this is the COMPLETED epoch number
        'epoch': epoch,
        
        # Training history
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        
        # Model config (for verification)
        'model_name': args.model,
        'learning_rate': args.learning_rate,
    }
    
    # Save checkpoint for this specific epoch
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as "latest" for easy resuming
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save weights-only for compatibility/inference
    weights_path = os.path.join(output_dir, f'epoch_{epoch + 1}.pt')
    torch.save(net.state_dict(), weights_path)
    
    print(f'✓ Saved checkpoint: epoch {epoch + 1}, LR={opt.param_groups[0]["lr"]:.2e}')
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, net, opt, sch, train_device):
    """
    Load COMPLETE checkpoint and restore all training state.
    
    Returns:
        start_epoch: Which epoch to start from (next epoch to train)
        train_losses, val_losses, train_accs, val_accs: Training history
    """
    if not os.path.exists(checkpoint_path):
        print(f'No checkpoint found at {checkpoint_path}, starting from scratch')
        return 0, [], [], [], []
    
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=train_device)
    
    # Check if it's the new format (with optimizer) or old format (weights only)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format: full checkpoint
        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        sch.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # The checkpoint contains the COMPLETED epoch, so we start from the next one
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])
        
        print(f'✓ Resumed from epoch {checkpoint["epoch"]} (completed)')
        print(f'  Will start training from epoch {start_epoch}')
        if train_accs:
            print(f'  Last train acc: {train_accs[-1]:.2f}%')
            print(f'  Last val acc: {val_accs[-1]:.2f}%')
        print(f'  Current LR: {opt.param_groups[0]["lr"]:.2e}')
        
        return start_epoch, train_losses, val_losses, train_accs, val_accs
    
    else:
        # Old format: weights only (your current saved files)
        print(' Warning: Loading weights-only checkpoint (no optimizer state)')
        print('  Training will continue but optimizer momentum is lost')
        net.load_state_dict(checkpoint)
        
        # Try to load training stats from JSON
        checkpoint_dir = os.path.dirname(checkpoint_path)
        stats_files = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('training_stats_') and f.endswith('.json')]
        
        if stats_files:
            latest_stats = max(stats_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            stats_path = os.path.join(checkpoint_dir, latest_stats)
            
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            start_epoch = stats.get('epochs_completed', 0)
            train_losses = stats.get('train_losses', [])
            val_losses = stats.get('val_losses', [])
            train_accs = stats.get('train_accs', [])
            val_accs = stats.get('val_accs', [])
            
            print(f'  Resuming from epoch {start_epoch}')
            if train_accs:
                print(f'  Last train acc: {train_accs[-1]:.2f}%')
            
            return start_epoch, train_losses, val_losses, train_accs, val_accs
    
    return 0, [], [], [], []


def main(args):
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        logging.info(f'Creating output directory {output_dir}...')
        os.makedirs(output_dir)

    # where to save/load the weights after training
    weights_path = os.path.join(output_dir, default_weights_fname)

    dataset = 4

    val_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/val_background_s24w6d1_1.hdf')
    val_npy = os.path.join(args.data_dir, f'dataset-{dataset}/v2/val_injections_s24w6d1_1.25s.npy')

    train_device = args.train_device

    base_model = get_model(args.model, train_device)
    norm = DAIN_Layer(input_dim=2).to(train_device)
    base_model.apply(initialize_xavier)

    net = CropWhitenNet(base_model, norm).to(train_device)

    validation_dataset = SlicerDataset(val_hdf, val_npy, slice_len=int(args.slice_dur * sample_rate),
                                       slice_stride=int(args.slice_stride * sample_rate),
                                       max_seg_idx=int(np.floor(args.slice_dur)))
    val_dl = DataLoader(validation_dataset, batch_size=100, shuffle=True, num_workers=args.num_workers,
                        pin_memory=train_device)

    background_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_background_s24w61w_1.hdf')
    injections_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_injections_s24w61w_1.hdf')
    inj_npy = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_injections_s24w61w_1.25s_all.npy')

    # Decode the SNR schedule once
    sch_epochs, sch_ranges = decode_snr_schedule(args.snr_schedule)
    
    # Initialize with first SNR range
    min_snr, max_snr = sch_ranges[0]
    training_dataset = SlicerDatasetSNR(background_hdf, inj_npy, slice_len=int(args.slice_dur * sample_rate),
                                        slice_stride=int(args.slice_stride * sample_rate),
                                        max_seg_idx=int(np.floor(args.slice_dur)),
                                        injections_hdf=injections_hdf, min_snr=min_snr, max_snr=max_snr,
                                        p_augment=args.p_augment)
    batch_size = args.batch_size
    train_dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=train_device)

    # setup loss
    loss = reg_BCELoss(dim=2)

    # setup optimizer
    learning_rate = args.learning_rate
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    milestones = [int(m) for m in args.lr_milestones.split(',')]
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.gamma)
    n_wrm = args.warmup_epochs
    wrm = WarmUpLR(opt, int(len(train_dl) * n_wrm))

    # Load checkpoint if resuming
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    if args.resume_from is not None:
        start_epoch, train_losses, val_losses, train_accs, val_accs = load_checkpoint(
            args.resume_from, net, opt, sch, train_device
        )
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
        print(f"{'='*60}\n")

    n_epochs = args.epochs
    
    # train/val loop - starts from start_epoch
    for epoch in range(start_epoch, n_epochs):
        
        print(f"\n{'='*60}")
        print(f"STARTING EPOCH {epoch} (out of {n_epochs})")
        print(f"{'='*60}")

        net.train()
        # train losses
        training_running_loss = 0.
        training_batches = 0
        # train accuracy
        total = 0
        correct = 0
        # val accuracy
        total_val = 0
        correct_val = 0

        # *** FIX: Use the actual epoch number for SNR schedule ***
        # This will correctly progress through the schedule even after resuming
        s_min, s_max = get_snr_by_epoch(sch_epochs, sch_ranges, epoch)
        training_dataset.set_snr_range(s_min, s_max)
        print(f"SNR range for epoch {epoch}: [{s_min}, {s_max}]")

        for idx, (training_samples, training_labels, training_inj_times) in enumerate(train_dl):
            training_samples = training_samples.to(device=train_device)
            training_labels = training_labels.to(device=train_device)

            # Optimizer step on a single batch of training data
            opt.zero_grad()

            training_output = net(training_samples, training_inj_times)
            training_loss = loss(training_output, training_labels)
            training_loss.backward()
            # Clip gradients to make convergence somewhat easier
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_norm)
            # Make the actual optimizer step and save the batch loss
            opt.step()

            # Warmup step after optimizer step (only if still in warmup period)
            if epoch < n_wrm:
                wrm.step()

            # get predictions & gt to measure accuracy
            _, predicted = training_output.max(1)
            _, gt = training_labels.max(1)
            total += training_output.size(0)
            correct += predicted.eq(gt).sum().item()
            train_acc = 100. * (correct / total)

            # update running loss
            training_running_loss += training_loss.clone().cpu().item()
            training_batches += 1

            progress_bar(idx, len(train_dl),
                         f'Epoch {epoch} | Loss {training_running_loss / training_batches:.2f} | Acc {train_acc:.2f}')

        # Evaluation on the validation dataset
        net.eval()
        with torch.no_grad():

            # error analysis
            positive_correct = 0
            positive_total = 0
            negative_correct = 0
            negative_total = 0

            val_predictions = []
            val_groundtruth = []

            validation_running_loss = 0.
            validation_batches = 0
            for val_idx, (validation_samples, validation_labels, validation_inj_times) in enumerate(val_dl):
                validation_samples = validation_samples.to(device=train_device)
                validation_labels = validation_labels.to(device=train_device)

                # Evaluation of a single validation batch
                validation_output = net(validation_samples, validation_inj_times)
                validation_loss = loss(validation_output, validation_labels)

                # get predictions & gt to measure accuracy
                _, predicted_val = validation_output.max(1)
                _, gt_val = validation_labels.max(1)
                total_val += validation_output.size(0)
                correct_val += predicted_val.eq(gt_val).sum().item()
                val_acc = 100. * (correct_val / total_val)
                validation_running_loss += validation_loss.clone().cpu().item()

                pos_idx = gt_val == 0
                neg_idx = ~pos_idx
                positive_total += pos_idx.sum()
                negative_total += neg_idx.sum()
                positive_correct += predicted_val[pos_idx].eq(gt_val[pos_idx]).sum().item()
                negative_correct += predicted_val[neg_idx].eq(gt_val[neg_idx]).sum().item()
                validation_batches += 1
                progress_bar(val_idx, len(val_dl),
                             f'Validation | Loss {validation_running_loss / validation_batches:.2f} | Acc {val_acc:.2f}'
                             f' (+:{100 * (positive_correct/positive_total):.3f}%,-:{100 * (negative_correct/negative_total):.3f}%)')

        # Print information and save
        validation_loss = validation_running_loss / validation_batches
        training_loss = training_running_loss / training_batches
        output_string = '%04i Train Loss: %f | Val Loss: %f || Train Acc: %.3f%% | Val Acc: %.3f%% (+:%.3f%%,-:%.3f%%)' % (
            epoch, training_loss, validation_loss,
            train_acc, val_acc, 100 * (positive_correct / positive_total), 100 * (negative_correct / negative_total))
        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        logging.info(output_string)
        
        # *** SAVE CHECKPOINT BEFORE SCHEDULER STEP ***
        # This ensures the checkpoint has the correct scheduler state for resuming
        save_checkpoint(
            epoch=epoch,
            net=net,
            opt=opt,
            sch=sch,
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            output_dir=output_dir,
            args=args
        )
        
        # *** STEP SCHEDULER AFTER SAVING ***
        # This updates the learning rate for the next epoch
        sch.step()
        print(f"Learning rate after epoch {epoch}: {opt.param_groups[0]['lr']:.2e}")
        
        # Also save legacy JSON stats for compatibility
        with open(os.path.join(output_dir, f'training_stats_{epoch + 1}.json'), 'w') as f:
            train_dict = {
                'model': args.model,
                'epochs_completed': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
            json.dump(train_dict, f, indent=2)

    # training over, save final network
    torch.save(net.state_dict(), weights_path)

    # training plots
    fig, axs = pylab.subplots(1, 2, sharex=True, figsize=(10, 5))
    fig.suptitle(f'Training loss & acc ({args.model.upper()})')
    axs[0].plot(train_losses, label='train')
    axs[0].plot(val_losses, label='val')
    axs[0].title.set_text('Loss')

    axs[1].plot(train_accs, label='train')
    axs[1].plot(val_accs, label='val')
    axs[1].title.set_text('Accuracy')

    fig.savefig(f'{output_dir}/training_curves.png')

    # Print final validation accuracy
    positive_acc = 100 * (positive_correct / positive_total)
    negative_acc = 100 * (negative_correct / negative_total)
    print(f'Validation accuracy: {val_acc}% (positive: {positive_acc}%, negative: {negative_acc}%)')

    # save final stats
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        train_dict = {
            'model': args.model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        json.dump(train_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    training_group = parser.add_argument_group('training')

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('-o', '--output-dir', type=str, help="Path to the directory where the outputs will be stored.")
    parser.add_argument('--data-dir', type=str, help='Path to the directory where data is stored.')
    parser.add_argument('--slice-dur', type=float, default=3.25, help='Duration (in s) of original slices.')
    parser.add_argument('--slice-stride', type=float, default=2., help='Slice stride.')

    training_group.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from.')
    training_group.add_argument('--learning-rate', type=float, default=5e-5, help="Learning rate.")
    training_group.add_argument('--lr-milestones', type=str, default='20,50', help='Epochs for LR decay.')
    training_group.add_argument('--gamma', type=float, default=0.5, help='LR decay rate.')
    training_group.add_argument('--epochs', type=int, default=10, help="Total number of training epochs.")
    training_group.add_argument('--snr-schedule', type=str, default='5:15-100,5:1-100', help='SNR schedule.')
    training_group.add_argument('--batch-size', type=int, default=32, help="Batch size.")
    training_group.add_argument('--warmup-epochs', type=float, default=0, help="Warmup epochs.")
    training_group.add_argument('--clip-norm', type=float, default=100., help="Gradient clipping norm.")
    training_group.add_argument('--p-augment', type=float, default=0.25, help="Augmentation probability.")
    training_group.add_argument('--train-device', type=str, default='cpu', help="Training device.")
    training_group.add_argument('--num-workers', type=int, default=8, help="DataLoader workers.")
    parser.add_argument('--model', type=str, default='resnet', choices=['cnn', 'resnet'], help="Model architecture.")

    args = parser.parse_args()

    main(args)