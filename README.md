![Potion_banner](https://github.com/if-loops/towards_poison_unlearning/assets/47212405/9578a0c1-9e65-4574-a71a-8afc2b705470)

# Potion: Towards Poison Unlearning

Our work is based on the benchmarking environment of [Corrective Machine Unlearning](https://github.com/drimpossible/corrective-unlearning-bench) 

## Abstract

Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. 
This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. 
Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. 
Our work addresses two key challenges to advance the state of the art in poison unlearning. 
First, we introduce a novel outlier-resistant SSD-based method to improve model protection and unlearning performance simultaneously.
Second, we introduce Poison Trigger Neutralisation (PTN) search. A fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100 with 0.2\%, 1\%, and 2\% of the data poisoned and discovery shares ranging from a single sample to 100\%.
Experimental results show that our method heals 93.72\% of poison compared to SSD with 83.41\% and full retraining with 40.68\%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68\% (SSD) to 1.41\% (ours).

![problem (1)](https://github.com/if-loops/towards_poison_unlearning/assets/47212405/e328df52-8af5-460e-9c49-85d7ea65428d)

## Using this repo
The .sh files are used to perform the experiments. Most parameters (e.g., $\rho$) can be changed directly in the .sh files, while others such as $s_{step}$ and $s_{start}$ are set in main.py and methods.py.
We have set up logging using weights and biases. Feel free to use alternative loggers.

## Citing this work

```
To be added
```

## Our related research

| Paper  | Code | Venue/Status |
| ------------- | ------------- |  ------------- |
| [Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening](https://arxiv.org/abs/2308.07707) | [GitHub](https://github.com/if-loops/selective-synaptic-dampening) |  AAAI 2024  |
| [ Loss-Free Machine Unlearning](https://arxiv.org/abs/2402.19308) (i.e. Label-Free) | [GitHub](https://github.com/if-loops/selective-synaptic-dampening) |  ICLR 2024 Tiny Paper  |
| [Parameter-Tuning-Free Data Entry Error Unlearning with Adaptive Selective Synaptic Dampening](https://arxiv.org/abs/2402.10098)  | [GitHub](https://github.com/if-loops/adaptive-selective-synaptic-dampening) |  Preprint  |
| [Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization](https://browse.arxiv.org/abs/2402.01401)  | [GitHub](https://github.com/jwf40/Zeroshot-Unlearning-At-Scale) |  Preprint  |
