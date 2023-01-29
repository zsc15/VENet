import os
import argparse
import torch
from networks.unet_model import UNet2, Unet, UNet, UNet2_with_contour, UNet_UAMT
from inference_one_img import calculate_imglist_metric as calculate2

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_root_path', type=str, default='../../datasets/Glas/testA/images/',
    #                     help='Name of Experiment')
    # parser.add_argument('--test_mask_path', type=str, default='../../datasets/Glas/testA/anno/', help='gt mask path')
    # parser.add_argument('--test_root_path', type=str, default='../../datasets/Glas/testB/images/', help='Name of Experiment')
    # parser.add_argument('--test_mask_path', type=str, default='../../datasets/Glas/testB/anno/', help='gt mask path')
    # parser.add_argument('--test_root_path', type=str, default='../../datasets/CRAG/test/images/',
    #                     help='Name of Experiment')
    # parser.add_argument('--test_mask_path', type=str, default='../../datasets/CRAG/test/anno/', help='gt mask path')
    parser.add_argument('--test_root_path', type=str, default='../../datasets/gland_images/images/',
                        help='Name of Experiment')
    parser.add_argument('--test_mask_path', type=str, default='../../datasets/gland_images/anno/', help='gt mask path')
    ###1_1_1UNet_ACWE_erosion_dice_ce, 1_1_1UNet_AC_loss, 1_1_1VENet_adam(dice+original model), 1_1_1UNet_ACWE_erosion_adam
    parser.add_argument('--model', type=str,  default='1.0_0.0_1.0_0.0VENet_adam_sigmoid_six_scales', help='model_name')
    ### 1_1_1VENet_adam 0.599998 for VENet test A, 0.55 for test B, 0.5 for CRAG
    ### 1.0_1.0_1.0_0.0VENet_adam_sigmoid_six_scales test A 0.594084
    ### 1.0_1.0_0.0_1.0VENet_adam_sigmoid_six_scales test A 0.590308
    ### 1.0_1.0_0.5_0.5VENet_adam_sigmoid_six_scales test A 0.594084
    ### 0.5_0.6_1.0_0.0VENet_adam_sigmoid_six_scales test A 0.594084
    parser.add_argument('--t', type=float, default=0.6, help='threshold for ACWE')
    parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
    parser.add_argument('--ori_dataset_type', type=bool,  default=False, help='images type')
    return parser.parse_args()

def create_model(ema=False):
    # Network definition
    net = UNet_UAMT(n_classes=num_classes)
    model = net.cuda(device.index)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

if __name__ == '__main__':
    args = get_args()
    img_testB_list = ['../../datasets/GlaS/testB_' + str(i + 1) + '.bmp' for i in range(20)]
    img_testB_anno_list = ['../../datasets/GlaS/testB_' + str(i + 1) + '_anno.bmp' for i in range(20)]
    img_testA_list = ['../../datasets/GlaS/testA_' + str(i + 1) + '.bmp' for i in range(60)]
    img_testA_anno_list = ['../../datasets/GlaS/testA_' + str(i + 1) + '_anno.bmp' for i in range(60)]
    # metric = calculate_glas_all_metric(net, device, img_testA_list, img_testA_anno_list, args)
    # metric1 = calculate_glas_all_metric(net, device, img_testB_list, img_testB_anno_list, args)
    # args.test_root_path = '../../datasets/GlaS_CRAG_cn_npy/testA400/images/'
    # args.test_mask_path = '../../datasets/GlaS_CRAG_cn_npy/testA400/label/'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    snapshot_path = "../model/{}".format(args.model)
    num_classes = 2
    if args.gpu == '2':
        device = torch.device('cuda:2')
    elif args.gpu == '1':
        device = torch.device('cuda:1')
    elif args.gpu == '3':
        device = torch.device('cuda:3')
    else:
        print('error!')
    # net = deeplabv3plus_resnet101(num_classes=1, pretrained_backbone=False).to(device)
    # test_save_path = os.path.join(snapshot_path, "test/")
    if args.model == 'UNet_ACWE_contour_erosion':
        net = UNet2_with_contour(n_classes=2).to(device)
        from inference_one_img import calculate_all_metric as calculate
    elif 'dice' or 'AC_loss' or 'VNet' in args.model:
        net = UNet2(n_classes=1, compute_sdm=False).to(device)
        from inference_one_img import calculate_all_metric as calculate
    elif 'SASSNet' in args.model:
        net = UNet2(n_classes=1, compute_sdm=True).to(device)
        from inference_one_img import calculate_all_metric as calculate
    elif 'npy' in args.model:
        args.test_root_path = '../../datasets/GlaS_CRAG_cn_npy/testA400/images/'
        args.test_mask_path = '../../datasets/GlaS_CRAG_cn_npy/testA400/label/'
        net = UNet2(n_classes=1, compute_sdm=False).to(device)
        from inference_one_img import calculate_all_metric_cn as calculate
    elif args.model == 'UNet_ours' or args.model == 'UNet_erosion' or 'ACWE' in args.model:
        net = UNet2(n_classes=1, compute_sdm=False).to(device)
        from inference_one_img import calculate_all_metric as calculate
    # elif args.model == 'UNet_contour':
    #     net = UNet(n_classes=1).to(device)
    #     from inference_one_img import calculate_net_one_channel_contour_metric as calculate
    elif args.model == 'UNet_cn':
        net = Unet(n_classes=1).to(device)
        from inference_one_img import calculate_net_one_channel_cn_metric as calculate
    elif args.model == 'UNet_cn2' and 'npy' in args.test_root_path:
        net = Unet(n_classes=1).to(device)
        from inference_one_img import calculate_net_one_channel_cn_metric2 as calculate
    elif args.model == 'Cg0.5_random_ACWE_arctan':
        net = UNet(n_classes=1).to(device)
        from inference_one_img import calculate_all_metric as calculate
    elif 'UAMT' in args.model:
        net = create_model().to(device)
        from inference_one_img import test_calculate_metric_two_channel as calculate
        # from inference_one_img import calculate_imglist_metric as calculate
    else:
        print('cao!')

    save_mode_path = os.path.join(snapshot_path, 'best.pth')
    net.load_state_dict(torch.load(save_mode_path))

    if args.ori_dataset_type:
        metric = calculate2(net, device, img_testB_list, img_testB_anno_list, args)

    else:
        metric = calculate(net, device, args) #6000
    print('average metric is #####\t obj-dice:%.5f, F1:%.5f, obj-hd:%.5f' %
          (metric[0], metric[1], metric[2]))
    # print('average metric is #####\t obj-dice:%.5f, F1:%.5f, obj-hd:%.5f' %
    #       (metric1[0], metric1[1], metric1[2]))
    # print('average metric of (dice, jc, hd, asd) is {}'.format(metric))
    # print(metric)