# Private Multi-Task Learning: Formulation and Applications to Federated Learning

This repository contains the code and experiments for the manuscript:

> [Private Multi-Task Learning: Formulation and Applications to Federated Learning](https://openreview.net/forum?id=onufdyHvqN)
>

Many problems in machine learning rely on *multi-task learning* (MTL), in which the goal is
to solve multiple related machine learning tasks simultaneously. MTL is particularly relevant
for privacy-sensitive applications in areas such as healthcare, finance, and IoT computing,
where sensitive data from multiple, varied sources are shared for the purpose of learning. In
this work, we formalize notions of client-level privacy for MTL via *billboard privacy* (BP), a
relaxation of differential privacy for mechanism design and distributed optimization. We
then propose an algorithm for mean-regularized MTL, an objective commonly used for
applications in personalized federated learning, subject to BP. We analyze our objective and
solver, providing certifiable guarantees on both privacy and utility. Empirically, we find that
our method provides improved privacy/utility trade-offs relative to global baselines across
common federated learning benchmarks.

This pytorch implementation is based off of the code from [Simplicial-FL repository](https://github.com/krishnap25/simplicial-fl) ([Laguel et al. 2021](https://ieeexplore.ieee.org/document/9400318)) and [Ditto repository](https://github.com/s-huu/Ditto) ([Li et al. 2020](https://arxiv.org/abs/2012.04221)).



## Preparation


### Prepare data

In order to run FEMNIST and CelebA experiment, please first go to ``` data/[dataset]/train ``` and ``` data/[dataset]/test ``` to unzip the .json file.

```
unzip *.json.zip
``` 

## Run on federated benchmarks

We provide scripts for StackOverflow(SO), FEMNIST, CelebA in `models`. Please see `run_[name_of_dataset].sh`.

(A subset of) Options in `models/run_[name_of_dataset].sh`:

* `sigma` indicates the standard deviation of the Gaussian Mechanism (Note this is *not* noise multiplier in [DP-SGD](https://arxiv.org/abs/1607.00133) paper).
* `gamma` indicates the clipping bound.
* `personalized` indicates whether we train the model with MTL or FedAvg.
* `start_finetune_rounds` indicates the number of epochs before performing finetuning.

