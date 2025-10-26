
###############################################################################
#
#   ELEC 475 - Lab 1, Jacob Chisholm
#   Fall 2025
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer


def main():

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-l', metavar='state', type=str,
                           help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size',
                           type=int, help='int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.l != None:
        save_file = args.l
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z
    else:
        bottleneck_size = 8

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True,
                      download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False,
                     download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(
        N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= train_set.data.size()[0]:
            print('label = ', train_set.targets[idx].item())
            img = train_set.data[idx]
            print('break 9', img.shape, img.dtype,
                  torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype,
                  torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype,
                  torch.min(img), torch.max(img))

            # plt.imshow(img, cmap='gray')
            # plt.show()

            img = img.to(next(model.parameters()).device)
            # print('break 7: ', torch.max(img), torch.min(img), torch.mean(img))
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(
                1, img.shape[0]*img.shape[1]).to(next(model.parameters()).device, dtype=torch.float32)
            print('break 9 : ', img.shape, img.dtype)
            with torch.no_grad():
                output = model(img)
            # output = output.view(28, 28).type(torch.ByteTensor)
            # output = output.view(28, 28).type(torch.FloatTensor)
            output = output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output),
                  torch.min(output), torch.mean(output))

            # plt.imshow(output, cmap='gray')
            # plt.show()
            #
            # both = np.hstack((img.view(28, 28).type(torch.FloatTensor),output))
            # plt.imshow(both, cmap='gray')
            # plt.show()

            img = img.view(28, 28).type(torch.FloatTensor)

            # Show side by side
            print("Showing reconstructed image")
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
            plt.show()

            # Prepare noisy image
            print("Showing denoise")
            img = train_set.data[idx]
            img = img.type(torch.float32)
            img = (img - torch.min(img)) / torch.max(img)
            # Add noise to the image
            img = img + (torch.rand(img.shape[0], img.shape[1]))
            # Send the image to the GPU if needed
            img = img.to(next(model.parameters()).device)
            img = img.view(
                1, img.shape[0]*img.shape[1]).to(next(model.parameters()).device, dtype=torch.float32)
            with torch.no_grad():
                output = model(img)
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            # Bring image back to 2D
            img = img.view(28, 28).type(torch.FloatTensor)
            output = output.view(28, 28).type(torch.FloatTensor)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
            plt.show()

            # Animation
            print("Preparing Linear Interpolation")
            idx2 = -1
            while idx2 < 0 or idx2 > train_set.data.size()[0]:
                idx2 = int(input("Enter a second idx > "))
            img1 = train_set.data[idx]
            img2 = train_set.data[idx2]
            img1 = img1.type(torch.float32)
            img2 = img2.type(torch.float32)
            img1 = (img1 - torch.min(img1)) / torch.max(img1)
            img2 = (img2 - torch.min(img2)) / torch.max(img2)
            # Send the image to the GPU if needed
            img1 = img1.to(next(model.parameters()).device)
            img2 = img2.to(next(model.parameters()).device)
            img1 = img1.view(
                1, img1.shape[0]*img1.shape[1]).to(next(model.parameters()).device, dtype=torch.float32)
            img2 = img2.view(
                1, img2.shape[0]*img2.shape[1]).to(next(model.parameters()).device, dtype=torch.float32)
            N_outputs = 10
            outputs = []
            outputs.append(img1.view(28, 28).type(torch.FloatTensor))
            with torch.no_grad():
                btl1 = model.encode(img1)
                btl2 = model.encode(img2)
                outputs.append(model.decode(btl1).view(
                    28, 28).type(torch.FloatTensor))
                for i in range(0, N_outputs):
                    interp = (1 - (i/N_outputs)) * btl1 + (i/N_outputs)*btl2
                    im = model.decode(interp)
                    outputs.append(im.view(28, 28).type(torch.FloatTensor))
                outputs.append(model.decode(btl2).view(
                    28, 28).type(torch.FloatTensor))
            outputs.append(img2.view(28, 28).type(torch.FloatTensor))
            N_outputs += 4
            f = plt.figure()
            for i in range(1, N_outputs+1):
                f.add_subplot(1, N_outputs, i)
                plt.imshow(outputs[i-1], cmap='gray')
            plt.show()


###################################################################
if __name__ == '__main__':
    main()
