template <typename Dtype>
void ConvDorefaLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConvDorefaParameter convDorefa_param = this->layer_param_.convolution_dorefa_param();
  const ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  containActive=convDorefa_param.contain_active();
  w_bit = convDorefa_param.w_bits();
  a_bit = convDorefa_param.a_bits();
  g_bit = convDorefa_param.g_bits();
  CHECK(w_bit>0);
  CHECK(a_bit>0);
  CHECK(g_bit>0);
  quanK2Pow_w=quanK2Pow_a=quanK2Pow_g=1.0;
  for(int i=0;i<w_bit && w_bit!=1;i++) quanK2Pow_w*=2.0;
  for(int i=0;i<a_bit && a_bit!=1;i++) quanK2Pow_a*=2.0;
  for(int i=0;i<g_bit && g_bit!=1;i++) quanK2Pow_g*=2.0;
  
  this->conv_learnable_blob_size=this->layer_param_.convolution_param().bias_term()==true?2:1;
  this->blobs_.resize(this->conv_learnable_blob_size);//fake
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_name(this->layer_param_.name() + "_internalConv");
    layer_param.set_type("Convolution");
    internalConv_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    internalConv_layer_->LayerSetUp(bottom,top);
    weightIntiByConv=false;
    scale_w=-1.;
    scale_a=-1.;
    blobsInitialized=false;
}

template <typename Dtype>
void ConvDorefaLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    internalConv_layer_->Reshape(bottom, top);

    //bitW.Reshape(internalConv_layer_->blobs()[0]->shape());
    if(containActive) bitA.Reshape(bottom[0]->shape());
    if(blobsInitialized==false)
    {
        if (conv_learnable_blob_size==2) {
          this->blobs_.resize(2);
        } else {
          this->blobs_.resize(1);
        }
        for(int i=0;i<this->conv_learnable_blob_size;i++)
        {
            this->blobs_[i].reset(new Blob<Dtype>(internalConv_layer_->blobs()[i]->shape()));
            caffe_copy(this->blobs_[i]->count(),internalConv_layer_->blobs()[i]->cpu_data(), this->blobs_[i]->mutable_cpu_data());
        }
        blobsInitialized=true;
    }

}


#ifdef CPU_ONLY
STUB_GPU(ConvDorefaLayer);
#endif

INSTANTIATE_CLASS(ConvDorefaLayer);
REGISTER_LAYER_CLASS(ConvDorefa);
