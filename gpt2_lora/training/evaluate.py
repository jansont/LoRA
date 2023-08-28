import time
import math
import torch
from gpt2_lora.utils import AverageMeter

def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()
    avg_t1_acc = AverageMeter()
    avg_all_acc = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            lm_logits, _loss, _t1_acc, _all_acc = model(
                _input, lm_labels=_target, lm_mask=_msk, is_report_accuracy=True) 
            t1_acc = _t1_acc.mean()
            all_acc = _all_acc.mean()
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())
            avg_t1_acc.update(t1_acc.item())
            avg_all_acc.update(all_acc.item())

            print('eval samples:', idx, 'loss:', loss.float(),
                      't1_acc:', t1_acc.float(), 'all_acc:', all_acc.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg), avg_t1_acc.avg, avg_all_acc.avg

