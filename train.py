"""
train.py
Training script for MoE-FFD + PhaseModule with optional JPEG augmentation.

Total Loss = CE Loss + MoE Load-Balancing Loss
"""
import os
import sys
sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import FFPP_Dataset, TestDataset
from utils import cal_metrics
from ViT_MoE_phase_aug import vit_base_patch16_224_in21k


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, optimizer, train_loader, valid_loader, scheduler, save_dir):
    max_accuracy = 0
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    os.makedirs(save_dir, exist_ok=True)

    if args.resume > -1:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models_params_{}.tar'.format(args.resume)),
            map_location='cuda:{}'.format(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(args.resume + 1, args.epochs):
        print('start train mode...')
        epoch_loss = 0.0
        total_num = 0
        correct_num = 0
        model.train()

        with torch.enable_grad():
            st_time = time.time()
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs, moe_loss, _, _, _, _ = model(inputs)

                ce_loss = criterion(outputs, labels)
                loss = ce_loss + moe_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_num += inputs.size(0)
                correct_num += torch.sum(torch.argmax(outputs, 1) == labels).item()
                global_step += 1

                if global_step % args.record_step == 0:
                    period = time.time() - st_time
                    train_acc = correct_num / total_num
                    train_loss = epoch_loss / (i + 1)
                    log.info('Training state: Epoch [{:0>3}/{:0>3}], Iter [{:0>3}/{:0>3}], '
                             'Loss: {:.4f} Acc:{:.2%} time:{}m {}s'
                             .format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                     train_loss, train_acc,
                                     int(period // 60), int(period % 60)))
                    writer.add_scalar('Loss/train', train_loss, global_step)
                    writer.add_scalar('Acc/train', train_acc, global_step)
                    st_time = time.time()
                    total_num = 0
                    correct_num = 0

        # Validation
        print('start eval mode...')
        model.eval()
        video_predictions, video_labels = [], []
        frame_predictions, frame_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, total=len(valid_loader), ncols=70, leave=False):
                inputs = inputs.cuda()
                inputs = inputs.squeeze(0)
                labels = labels.cuda()

                outputs, _, _, _, _, _ = model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                frame = outputs.shape[0]
                frame_predictions.extend(outputs[:, 1].cpu().tolist())
                frame_labels.extend(labels.expand(frame).cpu().tolist())
                pre = torch.mean(outputs[:, 1])
                video_predictions.append(pre.cpu().item())
                video_labels.append(labels.cpu().item())

        frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
        video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)
        log.info('valid result: Epoch [{:0>3}/{:0>3}], V_Acc: {:.2%}, V_Auc: {:.4} V_EER:{:.2%} '
                 'F_Acc: {:.2%}, F_Auc: {:.4} F_EER:{:.2%}'
                 .format(epoch + 1, args.epochs,
                         video_results.ACC, video_results.AUC, video_results.EER,
                         frame_results.ACC, frame_results.AUC, frame_results.EER))

        writer.add_scalar('AUC/val_video', video_results.AUC, epoch)
        writer.add_scalar('Acc/val_video', video_results.ACC, epoch)

        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(save_dir, 'models_params_{}.tar'.format(epoch)))

        if video_results.AUC > max_accuracy:
            for m in os.listdir(save_dir):
                if m.startswith('model_params_best'):
                    os.remove(os.path.join(save_dir, m))
            max_accuracy = video_results.AUC
            torch.save(model.state_dict(),
                       os.path.join(args.model_dir,
                                    'model_params_best_{:.4f}acc{:.4f}auc_epoch{:0>3}.pkl'.format(
                                        video_results.ACC, video_results.AUC, epoch + 1)))

        scheduler.step()

    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dv', type=int, default=0)
    parser.add_argument('--model_dir', '-md', type=str, default='models/train/')
    parser.add_argument('--resume', '-rs', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--record_step', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_freq_module', action='store_true', default=True)
    parser.add_argument('--uncertainty_routing', action='store_true', default=False)
    parser.add_argument('--entropy_thresh', type=float, default=1.0)
    parser.add_argument('--lambda_cons', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lambda_phase', type=float, default=0.5)
    parser.add_argument('--jpeg_aug', action='store_true', default=False,
                        help='Enable JPEG compression augmentation')
    parser.add_argument('--train_path', type=str, default='data/FF++/c23_crop/train',
                        help='Path to training data (real/ and fake/ subdirs)')
    parser.add_argument('--valid_path', type=str, default='data/FF++/c23_crop/val',
                        help='Path to validation data (real/ and fake/ subdirs)')
    args = parser.parse_args()

    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    save_dir = args.model_dir
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, 'train.log'),
        filemode='w' if args.resume == -1 else 'a',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    log.info('model dir: ' + args.model_dir)
    log.info('jpeg_aug: {}'.format(args.jpeg_aug))

    train_dataset = FFPP_Dataset(args.train_path, frame=20, phase='train', jpeg_aug=args.jpeg_aug)
    valid_dataset = TestDataset(args.valid_path, dataset='FFPP', frame=20)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1,
                              shuffle=False, num_workers=args.num_workers)

    model = vit_base_patch16_224_in21k(
        pretrained=True,
        num_classes=args.num_classes,
        use_freq_module=args.use_freq_module,
        uncertainty_routing=args.uncertainty_routing,
        entropy_thresh=args.entropy_thresh,
    )
    model = model.cuda()
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    special_param, other_param = [], []
    for name, param in model.named_parameters():
        if 'w_gate' in name or 'w_noise' in name:
            special_param.append(param)
        else:
            other_param.append(param)

    optimizer = optim.Adam(
        [{'params': special_param, 'initial_lr': 1e-4},
         {'params': other_param, 'initial_lr': args.learning_rate}],
        lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5, last_epoch=args.resume)

    print('Start train process...')
    train(args, model, optimizer, train_loader, valid_loader, scheduler, save_dir)

    duration = time.time() - start_time
    print('The run time is {}h {}m'.format(int(duration // 3600), int(duration % 3600 // 60)))
