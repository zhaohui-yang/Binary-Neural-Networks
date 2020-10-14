# Searching for Low-Bit Weights in Quantized Neural Networks (NeurIPS2020)

by Zhaohui Yang, Yunhe Wang, Kai Han, Chunjing Xu, Chao Xu, Dacheng Tao, Chang Xu. [[paper](https://arxiv.org/abs/2009.08695)] [[code](https://www.github.com/huawei-noah/slb)]

## Abstract 

Quantized neural networks with low-bit weights and activations are attractive for developing AI accelerators. However, the quantization functions used in most conventional quantization methods are non-differentiable, which increases the optimization difficulty of quantized networks. Compared with full-precision parameters (i.e., 32-bit floating numbers), low-bit values are selected from a much smaller set. For example, there are only 16 possibilities in 4-bit space. Thus, we present to regard the discrete weights in an arbitrary quantized neural network as searchable variables, and utilize a differential method to search them accurately. In particular, each weight is represented as a probability distribution over the discrete value set. The probabilities are optimized during training and the values with the highest probability are selected to establish the desired quantized network. Experimental results on benchmarks demonstrate that the proposed method is able to produce quantized neural networks with higher performance over the state-of-the-arts on both image classification and super-resolution tasks.

## VGG-Small and ResNet20 on CIFAR-10

```python
CUDA_VISIBLE_DEVICES=0 python3.5 train.py --arch=quan_vggsmall --dataset=c10 \
--scheduler=step --op_optimizer_method=sgd --op_learning_rate=0.02 --op_weight_decay=5e-4 \
--batch_size=100 --report_freq=100 --epochs=500 \
--basic_block_type=conv --w_nbit=1 --a_nbit=1
--temperature_speed=$temp --use_cr --bn_calibration \
--temperature_start_changing=0.3 --temperature_end_changing=0.9 
```

| VGG-Small (W/A) | temp | Reported | Epoch400 | Epoch500 |
| - | - | - | - | - |
| 1/1 | 1.03 | 92.0 | 92.0 | 92.4 |
| 1/2 | 1.03 | 93.4 | 93.3 | 93.3 |
| 1/4 | 1.03 | 93.5 | 93.6 | 93.8 |
| 1/8 | 1.03 | 93.8 | 93.6 | 93.8 |
| 1/32 | 1.04 | 93.8 | 93.7 | 93.7 |
| 2/2 | 1.04 | 93.5 | 93.6 | 93.7 | 
| 2/4 | 1.04 | 93.9 | 93.9 | 93.9 | 
| 2/8 | 1.04 | 94.0 | 93.9 | 94.0 | 
| 2/32 | 1.05 | 94.0 | 93.9 | 94.0 |
| 4/4 | 1.05 | 93.8 | 94.8 | 93.8 |
| 4/8 | 1.05 | 94.0 | 94.1 | 94.0 |
| 4/32 | 1.06 | 94.1 | 93.9 | 94.1 |

```python
CUDA_VISIBLE_DEVICES=0 python3.5 train.py --arch=quan_resnet20 --dataset=c10 \
--scheduler=step --op_optimizer_method=sgd --op_learning_rate=0.1 --op_weight_decay=1e-4 \
--batch_size=100 --report_freq=100 --epochs=500 \
--basic_block_type=convprelubn --w_nbit=1 --a_nbit=1 \
--temperature_speed=$temp --use_cr --bn_calibration --pact \
--temperature_start_changing=0.4 --temperature_end_changing=0.9 
```

| ResNet20 (W/A) | temp | Reported | Epoch400 | Epoch500 |
| - | - | - | - | - |
| 1/1 | 1.03 | 85.5 | 85.8 | 86.2 |
| 1/2 | 1.03 | 89.5 | 89.8 | 89.7 |
| 1/4 | 1.03 | 90.3 | 90.2 | 90.3 |
| 1/8 | 1.03 | 90.5 | 90.4 | 90.6 |
| 1/32 | 1.04 | 90.6 | 91.1 | 91.0 |
| 2/2 | 1.04 | 90.6 | 90.8 | 90.8 | 
| 2/4 | 1.04 | 91.3 | 91.0 | 91.5 | 
| 2/8 | 1.04 | 91.7 | 91.4 | 91.6 | 
| 2/32 | 1.05 | 92.0 | 92.1 | 91.8 |
| 4/4 | 1.05 | 91.6 | 91.4 | 91.5 |
| 4/8 | 1.05 | 91.8 | 92.0 | 92.0 |
| 4/32 | 1.06 | 92.1 | 92.1 | 92.2 |

## ResNet18 on ImageNet

```python
CUDA_VISIBLE_DEVICES=0 python3.5 train.py --arch=bireal_resnet18 --dataset=imagenet \
--op_optimizer_method=adam --op_learning_rate=1e-3 --op_weight_decay=0 \
--batch_size=256 --report_freq=500 --epochs=240 --scheduler=step \
--basic_block_type=convprelubn --w_nbit=1 --a_nbit=1 \
--temperature_speed=$temp --use_cr --bn_calibration --pact \
--temperature_start_changing=0.4 --temperature_end_changing=0.9 
```

| bireal\_resnet18 (W/A) | temp | Reported | This implementation |
| - | - | - | - |
| 1/1 | 1.05 | 61.3 | 61.5 |
| 1/2 | 1.05 | 64.8 | 64.9 |
| 1/4 | 1.05 | 66.0 | 66.5 |
| 1/8 | 1.05 | 66.2 | 66.8 |
| 1/32 | 1.05 | 67.1 | 66.9 |
| 2/2 | 1.05 | 66.1 | 66.7 |
| 2/4 | 1.05 | 67.5 | 68.0 |
| 2/8 | 1.05 | 68.2 | 68.4 |
| 2/32 | 1.05 | 68.4 | 68.6 |

## Tips

We have made some changes according to the constructive comments from NeurIPS reviewers. The changes/differences with paper and suggestions are as follows:

#### SBN Acceleration

The NeurIPS reviewer suggested that we could calibrate the BN statistics after training to accelerate training. However, the best model is not always achieved at last. To accelerate training, inspired by BigNAS and SPOSNAS, we calibrate the BN statistics after every epoch, the number of calibration iteration could be manually defined. In this implementation, we set the calibration iteration number the same as the length of validation loader. On ImageNet dataset, this change accelerates the calibration step by around 1280K(train set)/50K(val set) = 25 times.

#### Temperature changing by epoch

To avoid changing the temperature at each iteration, we change the temperature by exp scheduler after every epoch. The ```temperature_speed```, ```temperature_start_changing``` and ```temperature_end_changing``` hyper-parameters are used to control the temperature scheduler.

## Citation 

```
@inproceedings{yang2020searching,
	title="Searching for Low-Bit Weights in Quantized Neural Networks.",
	author="Zhaohui {Yang} and Yunhe {Wang} and Kai {Han} and Chunjing {Xu} and Chao {Xu} and Dacheng {Tao} and Chang {Xu}",
	booktitle="NeurIPS",
	year="2020"
}
```

