import numpy as np
import os
import glob
import cv2
import argparse
import warnings
import torch
from skimage.metrics import structural_similarity as ssim
from time import time

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

    model = cur_model.OPNet(sensing_rate=args.sensing_rate, LayerNo=args.layer_num)
    model = model.to(device)

    model_dir = "./{}/{}_group_{}_ratio_{:.2f}".format(args.save_dir, args.model, args.group_num, args.sensing_rate)
    checkpoint = torch.load("{}/net_params_{}.pth".format(model_dir, args.epochs), map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})

    ext = {'/*.jpg', '/*.png', '/*.tif'}
    filepaths = []
    test_dir = os.path.join('./DataSets/', args.test_name)
    for img_type in ext:
        filepaths = filepaths + glob.glob(test_dir + img_type)
    result_dir = "./{}/group_{}_ratio_{}/{}".format(args.result_dir, args.group_num, args.sensing_rate, args.test_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    Time_All = np.zeros([1, ImgNum], dtype=np.float32)

    with torch.no_grad():
        model(torch.zeros(1, 1, 256, 256).to(device))
        print("\nCS Reconstruction Start")
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]
            Img = cv2.imread(imgName, 1)
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:, :, 0]
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Img_output = Ipad / 255.
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_x = batch_x.unsqueeze(0).unsqueeze(0)

            start = time()
            x_output = model(batch_x)
            end = time()

            x_output = x_output[0]
            x_output = x_output.squeeze(0).squeeze(0)
            Prediction_value = x_output.cpu().data.numpy()
            X_rec = np.clip(Prediction_value[:row, :col], 0, 1)

            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)
            print("[{}/{}] Run time for {} is {:.4f}, PSNR is {:.2f}, SSIM is {:.4f}".format(
                img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

            Img_rec_yuv[:, :, 0] = X_rec * 255
            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

            resultName = "./{}/{}".format(result_dir, os.path.split(imgName)[1])
            cv2.imwrite("{}_CSratio_{}_epoch_{}_PSNR_{:.2f}_SSIM_{:.4f}.png".format(
                resultName, args.sensing_rate, args.epochs, rec_PSNR, rec_SSIM), im_rec_rgb)
            del x_output

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            Time_All[0, img_no] = end - start

    print('\n')
    output_data = "CS ratio is {}, Avg PSNR/SSIM/Time for {} is {:.2f}/{:.4f}/{:.4f}, Epoch number of model is {} \n".format(
        args.sensing_rate, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(Time_All), args.epochs)
    print(output_data)

    print("CS Reconstruction End")


def imread_CS_py(Iorg):
    block_size = args.block_size
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.math.log10(PIXEL_MAX / np.math.sqrt(mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='OPNet', help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.100000, help='set sensing rate')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--block_size', type=int, default=32, help='block size (default: 32)')
    parser.add_argument('--save_dir', type=str, default='save_temp', help='The directory used to save models')
    parser.add_argument('--group_num', type=int, default=200, help='group number for training')
    parser.add_argument('--layer_num', type=int, default=16, help='phase number of the Net')
    parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
    parser.add_argument('--result_dir', type=str, default='result', help='result directory')
    main()
