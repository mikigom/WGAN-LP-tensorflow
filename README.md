# WGAN-LP-tensorflow

[Report on arXiv](https://arxiv.org/abs/1712.05882)

Reproduction code for the following paper:

```
Title:	
On the regularization of Wasserstein GANs
Authors:	
Petzka, Henning; Fischer, Asja; Lukovnicov, Denis
Publication:	
eprint arXiv:1709.08894
Publication Date:	
09/2017
Origin:	
ARXIV
Keywords:	
Statistics - Machine Learning, Computer Science - Learning
2017arXiv170908894P
```
[Original Paper on arXiv](https://arxiv.org/abs/1709.08894)

## Repository structure

*data\_generator.py*
- provides a class that generates the sample data needed for learning.

*reg\_losses.py*
- defines the sampling method and loss term for regularization.

*model.py*
- implements 3-layer neural networks for a generator and a critic.

*trainer.py*
- a pipeline for model learning and visualization.
