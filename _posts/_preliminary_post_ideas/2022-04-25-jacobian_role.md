---
layout: article
title: Jacobian
key: A6
tags: Math
---

# Neural net Jacobians - And why they are important

Suppose a neural net computes an input output mapping $y = f_\phi(x)$. Typically $x \in \mathbb{R}^n$ and $y \in  \mathbb{R}^m$, thus $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$. Then the Jacobain matrix of this function is a $\mathbb{R}^{n \times m}$ matrix containing all partial derivatives

$$ J_f(a) = \left( \begin{aligned} &\frac{\partial f_1}{\partial x_1}(a) & \dots &\ \quad  \frac{\partial f_m}{\partial x_1}(a)\\ & \quad \vdots &  \ddots & \qquad \vdots\\  &\frac{\partial f_1}{\partial x_n}(a) & \dots &\ \quad  \frac{\partial f_m}{\partial x_n}(a) \end{aligned}  \right) $$

The reasone why it is import is that it defines the first order Taylor Approximation
$$ f(x + \delta) \approx f(x) + J_f(x)\cdot \delta$$

Thus if we want that our function does not change a lot into a particular direction given by some vector $\delta$, we have to bound
$$ ||J_f(x)\cdot \delta||$$

# TODO SOME LINEAR ALGEBRA

## Lipschitz continuity

Lipschitz continuity in easy terms: **Outputs should not change much in response to a small change to the inputs** i.e. a small change in the input produces a small change to the output.

> **Definition: (Lipschitz Continouty)**: A function $f$ is called L-Lipschitz continous in the (vector) norm $|| \cdot ||$$ if
> $$ ||f(x_1) - f(x_2)|| \leq L ||x_1 - x_2||$$

For conitnous univariate functions, we can formualte this a bit easier:

> **Lemma:** If $f$ is univariate and differentiable almost everywhere and $|f'(x)| \leq L$ almost everywhere then $f$ is L-Lipschitz continous.
> **Proof**: We will first proof the forward direction. Consider some $x_1 = x$ and $x_2 = x + h$. TODO PROOF IT

Fortunatly we can generalize this to the multivariate domain. Suppose $f(x) = Ax + b$ for some matrix $A$. Then 
$$ f \text{ is K-Lipschitz in} || \cdot || \iff ||A|| \leq K$$
where $||A||$ denote the matrix norm induced by the corresponding vector norm defined as
$$ ||A|| = \max_{||x|| = 1} ||Ax||$$
For example the standard euclidean 2-norm induces $||A||_2$ which equals the maximal singular value of $A$.

So the analogous lemma for the multivariate case is
> **Lemma:** If $f$ is multivariate and totally differentiable almost everywhere and $||J(x)|| \leq L$ almost everywhere then $f$ is L-Lipschitz continous.

Thus e.g. a function is L-Lipschitz in the Euclidean norm if the singular values of the Jacobian matrix is bounded by $L$. So why is this important? If i.e. a classification network is L-Lipschitz in some norm, then we can certify robustness agains $\epsilon$-ball pertubations. **Problem**: This may be to restrictive, bacuse $\epsilon$-balls are not a realistic threat model. It protects against pertubations in *any* directions, but we may just need to regularize the directions which e.g. reduce the loss the most! Further it imposes the constraint on any point, yet for some points it may be more important than for other (i.e. points near the decision boundary against points far away...).

Further Lipschitz-constrained functions are also relevant in other domian i.e. to compute the Wasserstein Distance!

### Practical estimation of the Jacobian norm

The size of the Jacobain is $n \times m$, which can be big for e.g. image classification tasks. In some cases it is easy:

* Scalar outputs:The jacobian equals the gradient, which can be estiamted by backprop.
* Many outputs: Hard to compute it explicitly, it requires $m$ forward and backward passes, which can be computed by backprop on each of the $m$ output dimensions. Yet if $m$ or $n$ is very large then this becomes very memory intesive... . Thus we typically access it through **implicit matrix-vector products**.

Notice that what backprop for a vector $v$ actually does is computing either $J_f(v)^T v$ using reverse mode automatic differentiation or $J_f(v)v$ i.e. forward mode automatic differentiation. Thus to compute the norm $||J_f||$ i.e. the largest singular value i.e. the largest eigenvalue of $J_f J_f^T$. We compute it with the power method!

### Jacobian for different layers

* Affine layer: $z = Wx$ then $J_{zx} = W$
* Elementwise operations: $y = exp(z)$ then $J_{yz} = diag(exp(z_1), \dots, \exp(z_d))$ 

As deep neural nets compose many of this layers, let's consider the Jacobian of composed functions which turns out to follow
$$ f(x) = h(g(x)) \text{ then } J_f = J_h J_g$$


### Connection to the Hessian

The Hessian is the second order derivative of $f$. Note in general it is only a matrix if $f: \mathbb{R}^n \rightarrow \mathbb{R}$ i.e. for functionals as e.g. the most loss functions in deep learning. In general it is an tensor! Yet in any way it can be thought as the local curvature. The problem is that if we have a neural network and want to compute the Hessian of the parameters then this will be HUGE. So instead in most deep learning libraries we use approximation i.e. the *Gauss-Newton approximation* i.e.
$$ H_\phi \approx J_{y\theta}^T H_y J_{y\theta}$$
This approximates the dependency of the prediction $y$ on the paramters $\theta$ as lienar (i.e. it drops the second order terms). Intuitivly if the loss curves rapodly in ap articular direection in parameter spaces, it is because the direction has a particulary large influence on the model's predicitons. Hence the second order Taylor approximation term must be large. This can be written as:
$$ v^T H_\theta v \approx (J_{y\theta} v)^T H_y (J_{y\theta} v)$$
notice that $J_{y\theta} v$ equals the sensitivity of the network in direction of $v$. 

Thus Jacobians are also relevant for the Curvature!



# Train neural networks subject to strict Lipschitz constraint





# References
https://www.broadinstitute.org/talks/primer-enforcing-lipschitz-constraints-neural-networks
