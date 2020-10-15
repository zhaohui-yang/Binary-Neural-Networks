# Training Binary Neural Networks through Learning with Noisy Supervision (ICML2020)

by Kai Han, Yunhe Wang, Yixing Xu, Chunjing Xu, Enhua Wu, Chang Xu. [[paper](https://arxiv.org/abs/2010.04871)] [[code](https://github.com/huawei-noah/Binary-Neural-Networks/tree/main/LNS)]

#### Abstract

This paper formalizes the binarization operations over neural networks from a learning perspective. In contrast to classical hand crafted rules (\eg hard thresholding) to binarize full-precision neurons, we propose to learn a mapping from full-precision neurons to the target binary ones. Each individual weight entry will not be binarized independently. Instead, they are taken as a whole to accomplish the binarization, just as they work together in generating convolution features. To help the training of the binarization mapping, the full-precision neurons after taking sign operations is regarded as some auxiliary supervision signal, which is noisy but still has valuable guidance. An unbiased estimator is therefore introduced to mitigate the influence of the supervision noise. Experimental results on benchmark datasets indicate that the proposed binarization technique attains consistent improvements over baselines.

## Citation 

```
@inproceedings{hantraining,
  title={Training Binary Neural Networks through Learning with Noisy Supervision},
  author={Han, Kai and Wang, Yunhe and Xu, Yixing and Xu, Chunjing and Wu, Enhua and Xu, Chang},
  booktitle={ICML},
  year={2020}
}
```