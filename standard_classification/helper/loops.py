from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F

from .util import AverageMeter, accuracy


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    entropies_temp = AverageMeter()  # Entropy with temperature scaling
    entropies_no_temp = AverageMeter()  # Entropy without temperature scaling

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            input = input.to(device)
            target = target.to(device)
            index = index.to(device)
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['itrd']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other distillation loss
        entropy_temp = None
        entropy_no_temp = None
        if opt.distill == 'kd':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'ab':
            loss_kd, entropy_temp, entropy_no_temp = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch)
        elif opt.distill == 'ttm':
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch)
        elif opt.distill == 'wttm':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'itrd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_correlation = opt.lambda_corr * criterion_kd.forward_correlation_it(f_s, f_t)
            loss_mutual = opt.lambda_mutual * criterion_kd.forward_mutual_it(f_s, f_t)
            loss_kd = loss_mutual + loss_correlation
        elif opt.distill == 'dist':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'ls':
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # Update entropy values if available
        if entropy_temp is not None:
            entropies_temp.update(entropy_temp.item(), input.size(0))
        if entropy_no_temp is not None:
            entropies_no_temp.update(entropy_no_temp.item(), input.size(0))

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Entropy_Temp {entropy_temp.val:.4f} ({entropy_temp.avg:.4f})\t'
                  'Entropy_NoTemp {entropy_no_temp.val:.4f} ({entropy_no_temp.avg:.4f})'.format(
                epoch, idx, len(train_loader), loss=losses, top1=top1, top5=top5,
                entropy_temp=entropies_temp, entropy_no_temp=entropies_no_temp))
            sys.stdout.flush()

    print(
        ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Entropy_Temp {entropy_temp.avg:.4f} Entropy_NoTemp {entropy_no_temp.avg:.4f}'
        .format(top1=top1, top5=top5, entropy_temp=entropies_temp, entropy_no_temp=entropies_no_temp))

    return top1.avg, losses.avg, entropies_temp.avg, entropies_no_temp.avg  # Return both entropies


def validate(val_loader, module_list, criterion_list, opt, is_Teacher=False):
    """validation"""

    # set modules as train()
    for module in module_list:
        module.eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_kd.eval()

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                input = input.to(device)
                target = target.to(device)
            index, pos_idx, neg_idx = None, None, None
            if is_Teacher:
                # compute output
                output = model_t(input)
                loss = criterion_cls(output, target)
            else:
                output = model_s(input)
                loss = criterion_cls(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
