#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e

EXAMPLE=cifar10
DATA=cifar-10-batches-bin
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

/media/hdd/lbl_trainData/git/caffe_DoReFa/build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

/media/hdd/lbl_trainData/git/caffe_DoReFa/build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
