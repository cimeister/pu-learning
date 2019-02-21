# Pytorch implementation of non-negative PU learning and unbiased PU learning
This is a reproducing code for non-negative PU learning [1] and unbiased PU learning [2] in the paper "Positive-Unlabeled Learning with Non-Negative Risk Estimator".

* ```loss.py``` has a pytorch implementation of the risk estimator for non-negative PU (nnPU) learning and unbiased PU (uPU) learning. 
* ```run_classifier.py``` is an example code of nnPU learning and uPU learning. 
Dataset is MNIST [3] preprocessed in such a way that even digits form the P class and odd digits form the N class.
The default setting is 1000 P data and 59000 U data of MNIST, and the class prior is the ratio of P class data in U data.

* Currently, the non-negative risk estimator does not seem to be working. If anyone wants to take a look and finds something wrong, please create a PR!!

## Requirements
* Python 3
* Torch >=1.0.1
* If using GPU, Cuda >=10.0

## Quick start
You can run an example code of MNIST for comparing the performance of nnPU learning and uPU learning on GPU.

    python run_classifier.py --data_dir=. --output_dir=model_file --do_train

You can see additional options by adding ```--help```.

## Example result


## Reference

[1] Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama. 
"Positive-Unlabeled Learning with Non-Negative Risk Estimator." Advances in neural information processing systems. 2017.

[2] Marthinus Christoffel du Plessis, Gang Niu, and Masashi Sugiyama. 
"Convex formulation for learning from positive and unlabeled data." 
Proceedings of The 32nd International Conference on Machine Learning. 2015.

[3] LeCun, Yann. "The MNIST database of handwritten digits." http://yann.lecun.com/exdb/mnist/ (1998).
