#!/bin/zsh
if [ -f "/bin/python" ]; then
    python train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
elif [ -f "/bin/python3" ]; then
    python3 train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
fi
