import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.unet_model import Unet
from utils import ramps
from inference_one_img import calculate_imglist_metric as calculate2
from utils.ACWE_loss import generalized_variational_Loss2 as Lgv
from utils.ACWE_loss import dice_loss
from dataloaders.ACWE_dataset import Gland_nanfang_img_npy_loader as gland_npy_set

parser = argparse.ArgumentParser()
parser.add_argument('--ori_dataset_type', type=bool, default=False, help='images type')
parser.add_argument('--root_path', type=str, default='../../datasets', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UNet_nanfang_Lv_npy', help='model_name')
parser.add_argument('--scale', type=list,  default=[1, 5/4, 1.5], help='scales of image')
parser.add_argument('--max_iterations', type=int,  default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4, help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--r', type=float,  default=0.5, help='labled rate')
parser.add_argument('--N', type=int,  default=258, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--t', type=float,  default=0.5, help='threshold for ACWE')
parser.add_argument('--threshold_type', type=str,  default='arctan', help='use a type of threshold function')
parser.add_argument('--random', type=bool,  default=False, help='randomly select parameters')
parser.add_argument('--alpha', type=float,  default=1, help='balance factor for ACWE')
parser.add_argument('--beta', type=float,  default=1, help='balance factor for sdm loss')
parser.add_argument('--mu', type=float,  default=1, help='the original ACWE model')
parser.add_argument('--nu', type=float,  default=0, help='the AC loss')
parser.add_argument('--gamma', type=float,  default=1, help='balance factor to control supervised and consistency loss')
parser.add_argument('--sigma', type=float,  default=0.1, help='balance factor to control supervised, adv and consistency loss')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=300.0, help='consistency_rampup')
#### distributed training
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
#### test dataset
parser.add_argument('--test_root_path', type=str, default='../../datasets/redo3/images/validation_npy/',
                    help='Name of Experiment')
parser.add_argument('--test_mask_path', type=str, default='../../datasets/redo3/annotations/validation_npy/',
                    help='gt mask path')
# parser.add_argument('--test_root_path', type=str, default='../../datasets/Glas/testA/images/',
#                     help='Name of Experiment')
# parser.add_argument('--test_mask_path', type=str, default='../../datasets/Glas/testA/anno/',
#                     help='gt mask path')
args = parser.parse_args()

train_data_path = args.root_path

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True #
    cudnn.deterministic = False #
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
# np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2

def arctan(t):
    return 0.5+2/(5*np.pi) * np.arctan(t/60000)

def sigmoid_like(t):
    return 0.6/(1+np.exp(-t/10000-np.log(5)))

def ladder(num):
    if 0 <= num <= 5000:
        t = 0.5
    elif 5000 < num <= 10000:
        t = (0.5+0.6)/2
    elif 10000 < num <= 15000:
        t = ((0.5 + 0.6) / 2+0.6)/2
    else:
        t = (((0.5 + 0.6) / 2+0.6)/2+0.6)/2
    return t

def update_learning_rate(schedulers, n):
    for scheduler in schedulers:
        scheduler.step(n)

def define_optimizer(model):
    G_optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            G_optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    G_optimizer = torch.optim.Adam(G_optim_params, lr=1e-4)
    return G_optimizer

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

if __name__ == "__main__":
    #####randomly select alpha, beta and gamma.
    if args.random:
        args.alpha = np.random.uniform()
        args.beta = np.random.uniform()
        args.gamma = np.random.uniform()
        args.consistency = np.random.uniform()
    snapshot_path = "../model/" + str(args.alpha)+'_'+str(args.beta)+'_'+str(args.mu)+'_'+str(args.nu)+args.exp + "/"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    #####
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.gpu == '0':
        device = torch.device('cuda:0')
    elif args.gpu == '1':
        device = torch.device('cuda:1')
    elif args.gpu == '2':
        device = torch.device('cuda:2')
    elif args.gpu == '3':
        device = torch.device('cuda:3')
    else:
        print('error!')
    # model = UNet2(n_classes=1)
    # #### train in one GPU
    # model = model.cuda(device.index)
    args.model = args.exp
    if 'npy' in args.test_root_path:
        net = Unet(n_classes=1)
        from inference_one_img import evaluate_nanfang as calculate
    else:
        print('please select a suitable model')
    img_testB_list = ['../../datasets/GlaS/testB_' + str(i + 1) + '.bmp' for i in range(20)]
    img_testB_anno_list = ['../../datasets/GlaS/testB_' + str(i + 1) + '_anno.bmp' for i in range(20)]
    img_testA_list = ['../../datasets/GlaS/testA_' + str(i + 1) + '.bmp' for i in range(60)]
    img_testA_anno_list = ['../../datasets/GlaS/testA_' + str(i + 1) + '_anno.bmp' for i in range(60)]
    ########initiation
    pretrained_model_path = '../model/1_1_1_0VENet_adam_npy/best.pth'
    model = net.cuda(device.index)
    # model.load_state_dict(torch.load(pretrained_model_path))
    ##############
    db_train = gland_npy_set(args, base_dir=train_data_path, mode='train') if 'npy' in args.model \
        else gland_set(args, base_dir=train_data_path)# train/val split

    db_test = gland_npy_set(args, base_dir=args.test_root_path, mode='test') if 'npy' in args.model \
        else gland_set(args, base_dir=train_data_path)  # train/val split
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = define_optimizer(model)
    schedulers = []
    schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000, 10000, 15000, 20000, 25000], 0.5))
    ce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()
    L1 = torch.nn.L1Loss()
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_dice = 0
    dict_img = 'norm' if 'npy' in args.model else 'image'
    dict_label = 'label'
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch[dict_img], sampled_batch[dict_label]
            volume_batch, label_batch = volume_batch.cuda(device.index), label_batch.cuda(device.index)
            model.train()
            # print(volume_batch.shape)
            x_seg = model(volume_batch)
            soft_seg = torch.sigmoid(x_seg)
            # print(soft_seg.shape, label_batch.shape)
            # loss_ACWE1, c_3, c_4 = Lgv(soft_seg, label_batch.float(), device, args, dim=0, threshold=args.t)
            loss_ACWE2, c_1, c_2 = Lgv(soft_seg[:, 0, ...], label_batch[:, 1, ...].float(), device, args,  threshold=args.t)
            loss_seg_dice = dice_loss(soft_seg, label_batch[:, 1:2, ...])
            loss_ce = ce_loss(x_seg, label_batch[:, 1:2, ...].float())### losses.dice_loss(outputs_soft[:labeled_bs, 1, ...], label_batch[:labeled_bs, 1, ...] == 1)
            if args.threshold_type =='arctan':
                args.t = arctan(iter_num)
            elif args.threshold_type =='ladder':
                args.t = ladder(iter_num)
            elif args.threshold_type == 'constant':
                args.t = 0.5
            elif args.threshold_type == 'sigmoid':
                args.t = sigmoid_like(iter_num)
            else:
                print('error!!!')
            loss = loss_seg_dice+args.alpha*loss_ce+args.beta*(loss_ACWE2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## Train D
            iter_num += 1
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_ACWE2: %f, '
                'c_1: %f, c_2: %f, threshold: %f ' %
                (iter_num, loss.item(), loss_ce.item(), loss_seg_dice.item(), loss_ACWE2.item(),
                 c_1.item(), c_2.item(), args.t))
            ## change lr
            update_learning_rate(schedulers, iter_num)
            if iter_num % 5000 == 0:
                # lr_ = base_lr * 0.5 ** (iter_num // 5000)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                model.eval()
                if args.ori_dataset_type:
                    metric = calculate2(model, device, img_testA_list, img_testA_anno_list, args)
                else:
                    # print(len(testloader))
                    metric = calculate(model, testloader, device)  # 6000
                    # print(len(testloader))
                # metric = calculate(model, device, args)
                # logging.info('average metric is #####\t obj-dice:%.5f, F1:%.5f, obj-hd:%.5f' %
                #             (metric[0], metric[1], metric[2]))
                logging.info('average metric is #####\t dice:%.5f' % metric)
                if metric >= best_dice:
                    best_dice = metric
                    # best_model_path = save_mode_path
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_wts_name = save_mode_path
                    # torch.save(model.state_dict(), best_model_path)
                model.train()
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    best_model_path = os.path.join(snapshot_path, 'best.pth')
    torch.save(best_model_wts, best_model_path)
    logging.info("best dice:%f" % best_dice)
    logging.info("best model:{}".format(best_model_wts_name))