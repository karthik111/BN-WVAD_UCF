import pdb
import numpy as np
import torch.utils.data as data
import utils
import time
import wandb
import torch

from options import *

from train import train
from losses import LossComputer
from test import test
from models import WSAD

from dataset_loader import *
from tqdm import tqdm
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"] = "true"


#import torch_xla
#import torch_xla.core.xla_model as xm

localtime = time.localtime()
time_ymd = time.strftime("%Y-%m-%d", localtime)
time_hms = time.strftime("%H:%M:%S", localtime)

if __name__ == "__main__":
    """_summary_
        args로부터 필요한 파라미터들을 받아오기
    """
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    args.log_path = os.path.join(args.log_path, time_ymd, 'ucf', args.version)
    args.model_path = os.path.join(args.model_path, time_ymd, 'ucf', args.version)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    
    wandb.init(
        project="BN-WVAD",
        name=args.version,
        config={
            'optimization:lr': args.lr[0],
            'optimization:iters': args.num_iters,
            'dataset:dataset': 'ucf-crime',
            'model:kernel_sizes': args.kernel_sizes,
            'model:channel_ratios': args.ratios,
            'triplet_loss:abn_ratio_sample': args.ratio_sample,
            'triplet_loss:abn_ratio_batch': args.ratio_batch,
        },
        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
    )


    worker_init_fn = None

    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    # plot_freq=5 seed가 다를 때의 실험을 위해 잠시 주석처리
    #device = xm.xla_device()
    #print(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = WSAD(args.len_feature,flag = "Train", args=args)
    #net = net.cuda()
    net = net.to(device)


    normal_train_loader = data.DataLoader(
        UCFVideo(root_dir = args.root_dir, mode = 'Train', num_segments = args.num_segments, len_feature = args.len_feature, is_normal = True),
            batch_size = args.batch_size,
            shuffle = True, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        UCFVideo(root_dir = args.root_dir, mode='Train', num_segments = args.num_segments, len_feature = args.len_feature, is_normal = False),
            batch_size = args.batch_size,
            shuffle = True, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCFVideo(root_dir = args.root_dir, mode = 'Test', num_segments = args.num_segments, len_feature = args.len_feature),
            batch_size = 10,
            shuffle = False, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn)


    test_info = {'step': [], 'AUC': [], 'AP': []}
    
    best_auc = 0

    criterion = LossComputer()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr[0],
        betas = (0.9, 0.999), weight_decay = args.weight_decay)

    best_scores = {
        'best_AUC': -1,
        'best_AP': -1,
    }

    metric = test(net, test_loader, test_info, 0)

    for step in tqdm(
            range(1, args.num_iters + 1),
            total = args.num_iters,
            dynamic_ncols = True
        ):
        ## 각 step 별 learning rate 및 dataloader 설정
        if step > 1 and args.lr[step - 1] != args.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
            
        ## 학습 및 loss 반환
        losses = train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion)
        wandb.log(losses, step=step)
        if step==0 or step == 300 or step == 700 or step==997:
            torch.save(net.state_dict(), os.path.join(args.model_path, f"wsad_epoch_{step}.pt"))
            print(f"Model saved at epoch {step}")
        ## 주기적으로 test를 통한 성능 확인
        if step % args.plot_freq == 0 and step > 0:
            metric = test(net, test_loader, test_info, step)
            print('AUC: ', test_info['AUC'][-1])
            if test_info["AUC"][-1] > best_scores['best_AUC']:
                utils.save_best_record(test_info, os.path.join(args.log_path, "ucf_best_record_{}.txt".format(args.seed)))

                torch.save(net.state_dict(), os.path.join(args.model_path, "ucf_best_{}.pkl".format(args.seed)))
            
            for n, v in metric.items():
                best_name = 'best_' + n
                best_scores[best_name] = v if v > best_scores[best_name] else best_scores[best_name]

        wandb.log(metric, step=step)
        wandb.log(best_scores, step=step)