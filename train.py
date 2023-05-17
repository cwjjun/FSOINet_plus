import argparse
import os
import warnings
import torch.optim as optim
import torch.nn.functional as F
from data_processor import *
from trainer import *
from scheduler import *

warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()
    if args.group_num == 100:
        import models.model100 as cur_model
    elif args.group_num == 200:
        import models.model200 as cur_model
    else:
        raise ImportError
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_dir = "./{}/{}_group_{}_ratio_{:.2f}".format(args.save_dir, args.model, args.group_num, args.sensing_rate)
    log_file_name = "./{}/{}_group_{}_ratio_{:.2f}.txt".format(model_dir, args.model, args.group_num, args.sensing_rate)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.backends.cudnn.benchmark = True

    model = cur_model.OPNet(sensing_rate=args.sensing_rate, LayerNo=args.layer_num)
    # model = nn.DataParallel(model)

    if args.sensing_rate != 0.5:
        pretrain_dir = "./{}/{}_group_{}_ratio_0.50".format(args.save_dir, args.model, args.group_num)
        checkpoint = torch.load("{]/net_params_100.pth".format(pretrain_dir))
        dict_new = model.state_dict().copy()
        new_list = list(model.state_dict().keys())
        dict_trained = checkpoint['net']
        trained_list = list(dict_trained.keys())
        # print("new_state_dict size: {}  trained state_dict size: {}".format(len(new_list), len(trained_list)))
        dict_new[ new_list[0] ] = dict_trained[ trained_list[0] ][:int(args.sensing_rate * 1024),:]
        for i in range(len(new_list) - 1):
            dict_new[new_list[i + 1]] = dict_trained[trained_list[i + 1]]
        model.load_state_dict(dict_new)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_epochs,
                                                                eta_min=args.flr)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_epochs,
                                           after_scheduler=scheduler_cosine)

    model = model.to(device)
    criterion = F.mse_loss
    train_loader = data_loader(args)

    if args.start_epoch > 0:
        pre_model_dir = model_dir
        checkpoint = torch.load("{}/net_params_{}.pth".format(pre_model_dir, args.start_epoch))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint["epoch"] + 1
        if args.sensing_rate == 0.5:
            for i in range(0, start_epoch):
                scheduler.step()
    else:
        start_epoch = args.start_epoch + 1
        if args.sensing_rate == 0.5:
            scheduler.step()

    print("Model: FSOINet+ , Sensing Rate: {:.2f} , Epoch: {} , Initial LR: {}\n".format(
        args.sensing_rate, args.epochs, args.lr))

    print('Start training')
    for epoch in range(start_epoch, args.epochs + 1):
        loss = train(train_loader, model, criterion, args.sensing_rate, optimizer, device)
        if args.sensing_rate == 0.5:
            print('current lr {:.5e}'.format(scheduler.get_lr()[0]))
            scheduler.step()
        print_data = "[{}/{}]Total Loss: {}".format(epoch, args.epochs, loss)
        print(print_data)
        output_file = open(log_file_name, 'a')
        output_file.write(print_data)
        output_file.close()

        # if epoch > 50 or args.sensing_rate != 0.5:
        #     if epoch % 5 == 0:
        #         checkpoint = {
        #             'epoch': epoch,
        #             'net': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #         }
        #         torch.save(checkpoint, "{}/net_params_{}.pth".format(model_dir, epoch))
        checkpoint = {
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, "{}/net_params_{}.pth".format(model_dir, epoch))


    print('Trained finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='OPNet', help='choose model to train')
    parser.add_argument('--sensing_rate', type=float, default=0.500000, help='set sensing rate')
    parser.add_argument('--group_num', type=int, default=100, help='group number for training')
    parser.add_argument('--start_epoch', default=0, type=int, help='epoch number of start training')
    parser.add_argument('--warm_epochs', default=3, type=int, help='number of epochs to warm up')
    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--block_size', default=32, type=int, help='block size (default: 32)')
    parser.add_argument('--lr', '--learning_rate', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--flr', '--final_learning_rate', default=5e-5, type=float, help='final learning rate')
    parser.add_argument('--save_dir', help='The directory used to save models', default='save_temp', type=str)
    parser.add_argument('--layer_num', type=int, default=16, help='phase number of the Net')
    main()
