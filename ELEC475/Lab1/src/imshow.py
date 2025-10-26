import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST


def main():
    print("Hello")
    train_transform: transforms.Compose = transforms.Compose(
        [transforms.ToTensor()]
    )
    train_set: MNIST = MNIST('./data/mnist', train=True,
                             download=True, transform=train_transform
                             )
    idx = -1
    while (idx < 0 or idx > 59999):
        idx = int(input("Input a number: "))
    plt.imshow(train_set.data[idx], cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
