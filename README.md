# uncertainty-quantification
Building a benchmark of uncertainty metrics and calibration curves for key deep learning uncertainty quantification methods


There has been limited deployment of deep neural networks in many safety-critical applications
due to their sensitivity to domain shifts and adversarial attacks, and their inability to provide
reliable uncertainty estimates for their decisions. In this work, a few key deep learning
techniques for uncertainty quantification were studied and implemented. These deep networks
were tested for their robustness to dataset perturbation and corruption, using CIFAR-10
rotations and the CIFAR-10-C dataset. In the process, a benchmark of uncertainty metrics and
calibration curves was built for these methods.


The log-likelihood and the Brier score metrics have been used to evaluate the uncertainty of
these models. The networks were based on the Resnet18 architecture. The networks can be
classified into:

1. Deterministic networks (Contrastive Reasoning, Test-time entropy minimization
(TENT)): These predict classes via a single forward pass.
2. Bayesian networks (Monte-Carlo Dropout, Bayes by Backprop, Bayes By Backprop
with the local reparameterization trick): Built on Bayesian principles, these require
multiple forward passes, with each pass giving a different output.
3. Ensemble networks (Bootstrap ensemble): These are a collection of several
deterministic networks, with the overall prediction being a combination of the predictions
of individual networks in the ensemble.

On CIFAR-10 rotations, the Bootstrap ensemble network performs the best, in terms of the brier
score and overall accuracy, whereas the Bayes by backprop technique outperforms the rest in
terms of the log-likelihood metric. Here, the variational inference (the two Bayes by backprop
networks) and the ensemble techniques are found to be better calibrated than the deterministic
networks, which are consistently underconfident with their predictions. On the CIFAR-10-C
dataset, the TENT technique performs the best, in terms of the brier score and accuracy, while
the bootstrap ensemble has the best log-likelihood score. The calibration performance is similar
to that on the previous dataset, with the deterministic networks performing worse than their
Bayesian and ensemble counterparts.

In the future, the uncertainty can be decomposed into aleatoric and epistemic uncertainties, to
find the root cause of uncertainty and improve the model accordingly. Another route would
involve improving the contrastive reasoning framework, by considering backpropagated
gradients from multiple layers as well as using Bayesian ideas for the custom loss function,
while still keeping the overall framework deterministic. One could also take inspiration from the
TENT technique and incorporate uncertainty estimates into the objective function during the
inference stage, and perform similar batch-wise model updates to improve model robustness.

## References
[1] Moloud Abdar et al. “A Review of Uncertainty Quantification in Deep Learning: Techniques,
Applications and Challenges”. In: arXiv (Nov. 2020). DOI: 10.1016/j.inffus.2021.05.008.
eprint: 2011.06225.
[2] Kaan Bıçakcı. “Uncertainty in Deep Learning — Epistemic Uncertainty and Bayes by Backprop”.
In: Medium (Apr. 2022). URL: https://towardsdatascience.com/uncertainty-in-deep-
learning-epistemic-uncertainty-and-bayes-by-backprop-e6353eeadebb.
[3] Charles Blundell et al. “Weight Uncertainty in Neural Networks”. In: ArXiv e-prints (May 2015).
DOI: 10.48550/arXiv.1505.05424. eprint: 1505.05424.
[4] Yarin Gal and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing Model
Uncertainty in Deep Learning”. In: ArXiv e-prints (June 2015). DOI:
10.48550/arXiv.1506.02142. eprint: 1506.02142.
[5] Dan Hendrycks and Thomas Dietterich. “Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations”. In: Proceedings of the International Conference on Learning
Representations (2019).
Georgia Tech Special Problem April 30, 2022 44 / 45
References
References II
[6] Diederik P. Kingma, Tim Salimans, and Max Welling. “Variational Dropout and the Local
Reparameterization Trick”. In: ArXiv e-prints (June 2015). DOI: 10.48550/arXiv.1506.02557.
eprint: 1506.02557.
[7] Mohit Prabhushankar and Ghassan AlRegib. “Contrastive Reasoning in Neural Networks”. In: ArXiv
e-prints (Mar. 2021). DOI: 10.48550/arXiv.2103.12329. eprint: 2103.12329.
[8] Dequan Wang et al. “Tent: Fully Test-Time Adaptation by Entropy Minimization”. In: International
Conference on Learning Representations. 2021. URL:
https://openreview.net/forum?id=uXl3bZLkr3c.

