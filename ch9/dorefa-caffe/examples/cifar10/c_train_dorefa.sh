#!/usr/bin/env sh
set -e

TOOLS=/media/hdd/lbl_trainData/git/caffe_DoReFa/build/tools

$TOOLS/caffe train \
  --gpu="0,1,2,3"  \
  --solver=/media/hdd/lbl_trainData/git/caffe_DoReFa/examples/cifar10/cifar10_dorefa_solver.prototxt $@
#--gpu="0,1,2,3"  \
#$TOOLS/caffe train \
#  --solver=examples/cifar10/cifar10_dorefa_solver.prototxt \
#  --snapshot=examples/cifar10/cifar10_dorefa_iter_4000.solverstate $@
