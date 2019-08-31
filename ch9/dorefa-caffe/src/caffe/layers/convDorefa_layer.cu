#include <vector>
#include <iostream>
#include "caffe/layers/convDorefa_layer.hpp"

namespace caffe {

using std::cout;
using std::endl;

template <typename Dtype>
void ConvDorefaLayer<Dtype>::binaryFw(Blob<Dtype>*fp, Blob<Dtype>*bin,const Dtype&bitCount)
{
    int N=fp->count();
    if(bitCount<1.01)
    {
        Dtype scale_=0;
        caffe_gpu_asum(N,fp->gpu_data(),&scale_);
        scale_/=N;
        caffe_gpu_quantizeK(N,fp->gpu_data(),bin->mutable_gpu_data(),(Dtype)1.0);
        caffe_gpu_scal(N,scale_,bin->mutable_gpu_data());
    }
    else
    {
        //cout<<"----------before caffe_gpu_clipByValue----------"<<endl;showDevice((float*)fp->gpu_data(),10);
        caffe_gpu_clipByValue(N,fp->gpu_data(), bin->mutable_gpu_data());
        //cout<<"----------caffe_gpu_clipByValue----------"<<endl;showDevice((float*)bin->gpu_data(),10);
        caffe_gpu_quantizeK  (N,bin->gpu_data(),bin->mutable_gpu_data(),bitCount-1);
        //cout<<"----------caffe_gpu_quantizeK----------"<<endl;showDevice((float*)bin->mutable_gpu_data(),10);
    }
}


template <typename Dtype>
void ConvDorefaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

        
    if(this->containActive)
    {
        caffe_gpu_copy(this->bitA.count(), bottom[0]->gpu_data(),this->bitA.mutable_gpu_data());
        binaryFw(bottom[0],bottom[0],this->quanK2Pow_a);
    }
    //cout<<"----------Forward_gpu weight----------"<<endl;showDevice((float*)this->blobs_[0]->gpu_data(),10);
    for(int i=0;i<this->conv_learnable_blob_size;i++)
        caffe_gpu_copy(this->blobs_[i]->count(), this->blobs_[i]->gpu_data(),internalConv_layer_->blobs()[i]->mutable_gpu_data());
    binaryFw(internalConv_layer_->blobs()[0].get(),internalConv_layer_->blobs()[0].get(),this->quanK2Pow_w);
    //cout<<"----------Forward_gpu 2----------"<<endl;showDevice((float*)this->blobs_[0]->gpu_data(),10);
    internalConv_layer_->Forward(bottom,top);
      //cout<<"----------conv_weight_quantity----------"<<endl;showDevice((float*)internalConv_layer_->blobs()[0]->gpu_data(),10);
      //cout<<"----------conv_diff----------"<<endl;showDevice((float*)internalConv_layer_->blobs()[0]->gpu_diff(),10);
}

template <typename Dtype>
void ConvDorefaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
              // cout<<"----------top_diff----------"<<endl;showDevice((float*)top[0]->gpu_diff(),10);
     internalConv_layer_->Backward(top,propagate_down,bottom);
          //caffe_gpu_set(this->blobs_[0]->count(), Dtype(2.0),this->blobs_[0]->mutable_gpu_diff());

    //cout<<"----------Forward_gpu weight----------"<<endl;showDevice((float*)this->blobs_[0]->gpu_data(),10);
    // cout<<"----------conv_diff----------"<<endl;showDevice((float*)internalConv_layer_->blobs()[0]->gpu_diff(),10);
    for(int i=0;i<this->conv_learnable_blob_size;i++)
    {
        caffe_gpu_copy(this->blobs_[i]->count(),internalConv_layer_->blobs()[i]->gpu_diff(), this->blobs_[i]->mutable_gpu_diff());
    }
    if(this->quanK2Pow_w>1.01)
    {
        caffe_gpu_clipByValue_grad(this->blobs_[0]->count(),this->blobs_[0]->gpu_diff(),this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff());
    }
    if(this->containActive)
    {
        if(this->quanK2Pow_a>1.01)
        {
        caffe_gpu_clipByValue_grad(bottom[0]->count(),bottom[0]->gpu_diff(),this->bitA.gpu_data(),bottom[0]->mutable_gpu_diff());
        }
    }
    //cout<<"----------Backward_gpu 3----------"<<endl;showDevice((float*)this->blobs_[0]->gpu_data(),10);
    //cout<<"----------Backward_gpu diff----------"<<endl;showDevice((float*)this->blobs_[0]->gpu_diff(),10);

}

INSTANTIATE_LAYER_GPU_FUNCS(ConvDorefaLayer);

}  // namespace caffe
