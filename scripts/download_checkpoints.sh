#!/bin/bash

# Utility function
function download_ckpt {
    FILEID=$1
    OUTNAME=$2

    out="$(wget \
        -q \
        --save-cookies /tmp/cookies.txt \
        --keep-session-cookies \
        --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" \
        -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')"


    wget \
        -q --show-progress \
        --load-cookies /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&confirm=${out}&id=$FILEID" -O $OUTNAME && rm -rf /tmp/cookies.txt
    
}

# Download checkpoints
CKPT_OUTPATH=./checkpoints
mkdir -p $CKPT_OUTPATH

echo "##################################################"
echo "# Downloading model checkpoints..."
echo "##################################################"
echo
download_ckpt "18cyKyX9UzKw3RP8RaVt7rdnGpt4nY3i7" "${CKPT_OUTPATH}/mnist.ckpt"
download_ckpt "1WQq4SZHpELIwB60O2A8LZ57f1dOhk3hf" "${CKPT_OUTPATH}/svhn.ckpt"
download_ckpt "1Ly5bA_4fZNLrizS0lxGNTDc5kkd85SvP" "${CKPT_OUTPATH}/cifar10.ckpt"
download_ckpt "1VzAeRbT9ALTy4EcJhKki5TlcLVkqEwFo" "${CKPT_OUTPATH}/cifar+10.ckpt"
download_ckpt "14j_IUiKaY4lx8OapLRSjDIgA3jlqSpiO" "${CKPT_OUTPATH}/cifar+50.ckpt"
download_ckpt "1aasfPvasXWJ35QRyRzChzBxtKxm6GLc0" "${CKPT_OUTPATH}/tiny_imagenet.ckpt"
echo
echo "##################################################"
echo "# All done! Model checkpoints downloaded in ${CKPT_OUTPATH}."
echo "##################################################"
