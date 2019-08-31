#ifndef CAFFE_CONVDOREFA_LAYER_HPP_
#define CAFFE_CONVDOREFA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class ConvDorefaLayer : public Layer<Dtype> {
 public:
  explicit ConvDorefaLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ConvDorefa"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    shared_ptr<Layer<Dtype> > internalConv_layer_;
    bool containActive;  
    bool weightIntiByConv;
    int w_bit;
    int a_bit;
    int g_bit;
    int conv_learnable_blob_size;
    Blob<Dtype> bitW;
    Blob<Dtype> bitA;
    Blob<Dtype> bitG;
    Dtype scale_w;
    Dtype scale_a;
    Dtype quanK2Pow_w;
    Dtype quanK2Pow_a;
    Dtype quanK2Pow_g;
    bool blobsInitialized;
    void binaryFw(Blob<Dtype>*fp, Blob<Dtype>*bin,const Dtype&bitCount);
};

}  // namespace caffe

#endif  // CAFFE_SPP_LAYER_HPP_
