template <typename Dtype>
class CConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit CConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "CConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  
 private:
  Blob<Dtype> weight_tmp_;
  Blob<Dtype> bias_tmp_;   
  Blob<Dtype> rand_weight_m_;
  Blob<Dtype> rand_bias_m_;  
  Dtype gamma,power;
  Dtype crate; 
  Dtype mu,std;    
  int iter_stop_;
};
