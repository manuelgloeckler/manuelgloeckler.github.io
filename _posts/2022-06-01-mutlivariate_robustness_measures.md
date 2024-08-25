---
layout: article
title: Local Variations in Functions
tags: Mathematics
aside:
    toc: true
cover: https://upload.wikimedia.org/wikipedia/commons/5/58/Lipschitz_Visualisierung.gif
comment: true
---

In this article, we delve into the intricacies of local variations in functions. We extend the concept of Lipschitz functions from univariate to multivariate functions and establish a connection with the operator norm of the Jacobian matrix. Additionally, we explore how different norms can yield diverse interpretations of Lipschitzness, offering versatile tools for analyzing function variations.

## Ensuring Worst-Case Guarantees
### Univariate Case

In the one-dimensional realm, where we have a function $f: \mathcal{X} \rightarrow \mathcal{X}$, the local variation of a function is quantified by its derivative i.e. it's instantaneous rate of change. If the absolute slope is bounded by $L$ within a set $S \subset \mathcal{X}$, it follows that $ \mid f(x) - f(y) \mid \leq L \cdot \mid x-y ~\mid$ for all $x, y \in S$. 

To demonstrate the reverse direction, consider an $x, y \in S$ assuming without restriction of generality that $y = x + h$. It is easy to see that 

$$ \mid f(x) - f(x + h) \mid \leq L \cdot h \iff \frac{\mid f(x) - f(x+ h) \mid}{h} \leq L \iff \lim_{h\rightarrow 0} \frac{\mid f(x) - f(x+ h) \mid}{h} \leq L  \iff \mid f'(x) \mid \leq L $$

Conversely, if $ \mid f'(x) \mid \leq L$ and $x, y \in S$, then by the mean value theorem, there exists $\xi \in (x, y)$ such that

$$ \mid f'(\xi) \mid = \frac{\mid f(x)-f(y) \mid}{\mid x-y \mid} \iff \mid f(x)-f(y) \mid = \mid f'(\xi) \mid \cdot \mid x-y \mid \iff \mid f(x)-f(y) \mid \leq L \cdot \mid x-y \mid $$

This is known as a *local Lipschitz* property. In this article, we investigate this property in the context of multivariate functions and explore alternative methods for quantifying local function variations.

### Multivariate Case

We consider a completely differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$. The first-order derivative is represented by the Jacobian matrix $J_f(x) \in \mathbb{R}^{n \times m}$. Analogously, we define the *local Lipschitz* property as follows:

<div class="definition">
A function $f$ and a set $S \subset \mathcal{X}$ are termed locally L-Lipschitz for $L \geq 0$ in the (vector) norm $ \mid  \mid  \cdot  \mid  \mid $ if for all $x, y \in S$,

  $$  \mid  \mid f(x) - f(y) \mid  \mid  \leq L  \mid  \mid x - y \mid  \mid $$
</div>

Notably, the first-order derivative is now a matrix, necessitating the concept of a matrix or operator norm. Given two vector norms $ \mid  \mid  \cdot  \mid  \mid_a$ and  $\mid  \mid \cdot \mid  \mid_b$, the induced operator norm of a linear map represented by a matrix $A$ is defined as

$$  \mid  \mid  A  \mid  \mid = \sup_{ \mid  \mid  x  \mid  \mid _a = 1}  \mid  \mid  Ax \mid  \mid _b$$

For example, the spectral norm is induced by Euclidean vector norms and can be computed as

$$  \mid  \mid A \mid  \mid _2 = \sqrt{ \lambda_{max}(A^T A)} =  \sigma_{max}(A) $$

where $\sigma_{max}$ denotes the largest singular value of $A$.

However, relating the Lipschitz property with the Jacobian operator norm is not straightforward. We require a generalization of the mean value theorem to the multivariate case.

<div class="lemma">
For a differentiable function $f$ on an open set $S \subset \mathbb{R}^n$, consider $x, y \in S$, and let $L_{xy}$ be the line segment connecting them. If $L_{xy}$ lies entirely within $S$, then there exists an $\xi \in L_{xy}$ such that
$$ f(x) - f(y) = J_f(\xi)(x-y)$$ 
</div>
[proof]
The line segment is given by $c(t) = tx + (1-t)y$ for $t \in [0,1]$. Consider the one-dimensional function $g(t) = f(c(t))$, then by the standard mean value theorem there exists some $\tau$ such that $$ g(1) - g(0) = (1-0) g'(\tau) \iff f(x) -f(y) = J_f(c(\tau))(x-y)$$
[/proof]

This result, while straightforward, is somewhat restrictive as it only applies to linear paths between points fully contained in $S$. We can ensure these constraints by stipulating that $S$ is convex. A set is considered convex if, for all $x, y \in S$ and $\lambda \in [0,1]$, the point $\lambda x + (1-\lambda)y$ also belongs to $S$, which is a requirement for our proof to hold. We can now state a more general theorem known as the "multivariate mean value theorem."

<div class="lemma">
For a differentiable function $f$ on an open and convex set $S \subset \mathbb{R}^n$, then for any $x, y \in S$, we have
$$  \mid  \mid f(x) - f(y) \mid  \mid _a \leq  \mid  \mid J_f(\xi) \mid  \mid \cdot  \mid  \mid x-y \mid  \mid _b$$ 
</div>
[proof]
By Lemma 1 we know that for each fixed $x, y \in S$ there is an $\xi \in S$ such that
$$ f(x) - f(y) = J_f(\xi) (x-y) \iff  \mid  \mid  f(x) - f(y)  \mid  \mid _a =  \mid  \mid  J_f(\xi) (x-y)  \mid  \mid _a$$
By the fundamental inequality of operator norm
$$  \mid  \mid  J_f(\xi) (x-y)  \mid  \mid _a \leq  \mid  \mid  J_f(\xi) \mid  \mid   \mid  \mid  (x-y)  \mid  \mid _b$$
which proofs the statement.
[/proof]


This theorem allows us to establish the local Lipschitz criteria.

<dir class="theorem">
A function $f$ is locally L-Lipschitz on an open and convex set $S$ in the norm $ \mid  \mid \cdot  \mid  \mid _a$ and $ \mid  \mid \cdot \mid  \mid _b$ if and only if 
 
 $$ \forall x \in S:  \mid  \mid J_f(x) \mid  \mid  \leq L$$
 where the operator norm is induced by the vector norms $ \mid  \mid \cdot  \mid  \mid _a$ and $ \mid  \mid \cdot \mid  \mid _b$. This further implies that the norm of all directional derivatives $ \mid \mid f_v'(x) \mid \mid_a$ is bounded by $L$.
</dir>
[proof]
Using the multivariate mean value theorem we have that for all $x, y \in S$ <br>
$$  \mid  \mid  f(x) - f(y)  \mid  \mid _a\leq  \mid  \mid  J_f(\xi) \mid  \mid  \cdot \mid  \mid  (x-y)  \mid  \mid _b \leq L  \mid  \mid (x-y) \mid  \mid _b$$ <br>
Which proves the reverse direction.<br>
<br>
Now if $f$ is locally L-Lipschitz then for some $x \in S$ and $y = x + \lambda v \in S$ where $ \mid  \mid v \mid  \mid _b = 1$ we have that
$$  \mid  \mid  f(x) - f(y)  \mid  \mid _a \leq L  \mid  \mid (x-y) \mid  \mid _b \iff \frac{  \mid  \mid  f(x) - f(x + \lambda v)  \mid  \mid _a} {  \mid \lambda} \leq L$$<br>
Note that by taking the limit $ \lambda \rightarrow 0$ we obtain the directional derivative of $f$ in direction $v$, denoted as $ \mid \mid f_v'(x)\mid \mid_a$. This implies that 
$ f_v'(x) \leq L \text{ for any direction } v$ <br>
Thus it follows that <br>
$$ \begin{aligned}
 \mid  \mid J_f(x) \mid  \mid  &= \sup_{ \mid  \mid v \mid  \mid _b = 1}  \mid  \mid  J_f(x) v \mid  \mid _a \\ 
&= \sup_{ \mid  \mid v \mid  \mid _b = 1}  \mid  \mid f_v'(x) \mid  \mid _a\\ 
&= L
\end{aligned}
$$
[/proof]


### Applications

This notion of local Lipschitzness is valuable for various applications, such as constraining the Lipschitz constant in neural networks. A neural network can often be expressed as a composition of linear maps and nonlinear activation functions in a "layer-wise" manner. For a single-layer ReLU network, the chain rule reveals the Jacobian's form:

$$ f(x) = g_1(W_1x + b)$$

The Jacobian for this network can be expressed as:

$$ J_f(x) = J_{g}(h_1)W_1, \text{ where } h_1 = Wx + b $$

Here, $J_{g}$ denotes the Jacobian of the ReLU nonlinearity. It takes on the form of a diagonal matrix that consists of zeros and ones, depending of the input to it is positive or negative.

To compute the local Lipschitz constant for a given convex set $S$, we can solve the following optimization problem:

$$ L = \sup_{x \in S} \mid \mid  J_f(x)\mid \mid  $$

Solving this optimization problem can be challenging, as the Jacobian depends on the hidden activations, necessitating constrained numerical optimization techniques. Yet, we can find an upper bound more easily by leveraging the submultiplicative property of the operator norm:

$$ L = \sup_{x \in S} \mid \mid  J_f(x)\mid \mid  = \sup_{x \in S} \mid \mid  J_{ReLU}(h(x))W\mid \mid  \leq  \sup_{x \in S} \mid \mid  J_{ReLU}(h(x))\mid \mid  \cdot \mid \mid W\mid \mid  \leq \mid \mid W\mid \mid  $$

This upper bound on the Lipschitz constant can be for example be enforced by constraining the operator norm of the network weights. For instance, we can normalize the spectral norm, which is the largest singular value of the weight matrix. This is a common technique for ensuring robustness in neural networks. But recall that this implicitly assumes that the right measure of distance within the input and output space is the Euclidean norm. 
## Probabilistic interpretation

As one can guess, the above statements can be interpreted as "worst-case" guarantees. This simply follows by definition of operator norms: It equals the worst-case absolute slope in any possible direction. Yet, sometimes we may be interested in average-case guarantees.

Be $p(v)$ a uniform distribution on a unit-sphere induced by some norm $\mid \mid \cdot \mid \mid _a$. For example, in the case of the L2 norm, we have that $v \sim p(v) \iff v = \frac{w}{\mid \mid w\mid \mid _2}$ with $w \sim \mathcal{N}(0, I)$. We already encountered the notion of directional derivatives in the previous theorem. Bounding the directional derivatives will also bound the local Lipschitz constant. Thus let's investigate its distribution!

A prior $p(v)$ on a unit-sphere induces a distribution on the directional derivatives as $f_v'(x) = J_f(x)v$. This distribution will have interesting properties. Let's again consider the L2 norm and compute the first and second moments. Notice that given a uniform distribution on the L2 unit sphere, we have that $$\mathbb{E}_{p(v)}[v] = 0$$ and $$\Sigma_{p(v)}[v] = \frac{1}{d}I$$ thus

$$ \mathbb{E}_{p(v)}[f_v'(x)] = J_f(x)\mathbb{E}_{p(v)}[v] = 0$$

$$ \Sigma_{p(v)}[f_v'(x)] = \mathbb{E}_{p(v)}[ J_f(x)vv^TJ_f(x)^T] = \frac{1}{d} J_f(x)J_f(x)^T $$

So this is a nice closed-form solution and again involves the Jacobian matrix. Recall these results are induced by the L2 norm, so they also should be related to the operator norm induced by the L2 norm. Recall that $\mid \mid  J_f(x)\mid \mid $ equals the largest singular value and thus is related to the eigenvalue of $J_f(x)^T J_f(x)$. Be $v$ the corresponding eigenvector then

$$ J_f(x)^T J_f(x)v = \lambda v \iff J_f(x)J_f(x)^T(J_f(x)v) = \lambda (J_f(x)v)$$

and hence both share the same eigenvalues. As $J_f(x)J_f(x)^T$ is up to a constant the covariance matrix of directional derivatives, the eigenvector with the largest eigenvalue hence points in the direction of the largest variation as required by the definition of the operator norm.

Yet for example if we instead want to control the average variation of the function, this suggests controlling all the eigenvalues i.e. by controlling the trace of the covariance.


## Investigating other norms

Both the operator norm as well as the covariance matrix of the directional derivatives can be hard to compute for non-standard norms. We will investigate here the problem in the case of a more general "euclidean-like" norm, the *elliptical norm*.

Why would we be interesting in doing so? Well, the inputs and outputs of e.g. a neural networks are typically not Euclidean vectors. For example, the inputs may be images and The outputs may be probability distributions. Images are invariant to translations and rotations and thus the Euclidean norm is not a good measure of distance. Similarly, probability distributions are not vectors and thus the Euclidean norm is not a good measure of distance. Controlling the Lipschitz constant in the Euclidean norm is thus not a good way to enforce a robustness property. Instead, we may want to control the Lipschitz constant in a norm that is more suitable for the task at hand.

<dir class="definition">
 An elliptical (pseudo-)norm $\mid \mid  \cdot \mid  \mid _S$ is defined for a positive (semi) definite matrix $S$ as given by

  $$  \mid  \mid x \mid  \mid_S = \sqrt{x^T S x} $$
</dir>

It is straightforward to verify that this is indeed a norm. A well-known special case is the *Mahalanobis distance*. We call it *elliptical* as the vectors of norm one lie on an ellipse spanned by the eigenvectors of $S$.

### The induced operator norm

The corresponding operator norm induced by $\mid \mid \cdot\mid \mid_S$ and $\mid \mid \cdot\mid \mid_2$ is given by

$$ \mid \mid  A \mid \mid_{S2}  = \sup_{\mid \mid v\mid \mid _S = 1} \mid \mid Av\mid \mid _2 = \sup_{\mid \mid v\mid \mid _S^2 = 1} \mid \mid Av\mid \mid _2^2$$

Thus we have to solve a constrained optimization problem. Let's try to solve it using Lagrange multipliers. The corresponding Lagrangian can be written as

$$ \mathcal{L}(v, \lambda) = v^T A^T A v - \lambda (v^T S v - 1) $$

Taking the derivative with respect to $v$ and $\lambda$ leads to

$$ \nabla_v \mathcal{L}(v, \lambda) = 2 A^T A v - \lambda 2 Sv$$

$$ \nabla_\lambda \mathcal{L}(v, \lambda) = v^T S v - 1$$

Setting them equal to zero yields

$$ 2 A^T A v - \lambda 2 Sv = 0 \iff A^T A v = \lambda Sv \quad \text{ and } \quad  v^T S v = 1 $$

Which is known as the generalized eigenvalue problem. Thus the operator norm is given by the maximal generalized singular value.

Now consider the other way around, i.e., the operator norm induced by  $\mid \mid \cdot\mid \mid_2$ and $\mid \mid \cdot\mid \mid_S$:

$$ \mid \mid  A \mid \mid_{2S}  = \sup_{\mid \mid v\mid \mid_2 = 1} \mid \mid Av\mid \mid_S = \sup_{\mid \mid v\mid \mid_2^2 = 1} \mid \mid Av\mid \mid _S^2$$

Similarly, the Lagrangian is given by

$$ \mathcal{L}(v, \lambda) = v^T A^T S A v - \lambda (v^T v - 1) $$

Taking the derivative with respect to $v$ and $\lambda$ leads to

$$ \nabla_v \mathcal{L}(v, \lambda) = 2 A^TS A v - \lambda 2v$$

$$ \nabla_\lambda \mathcal{L}(v, \lambda) = v^T v - 1$$

Setting this to zero leads to the solution

$$ A^T S A v = \lambda v \text{ with }  \mid  \mid v \mid  \mid _2^2 = 1 $$

which is a standard eigenvalue problem on a slightly different matrix $A^T S A$.

It is straightforward to see that relying only on $\mid \mid \cdot \mid \mid_S$ can also be stated as a generalized eigenvalue problem on this matrix.

### The covariance of the directional derivatives

We can change the covariance matrix by changing the "prior" $p(v)$. Our goal may be again to draw uniformly from the unit-ellipse induced by the norm $ \mid  \mid\cdot \mid  \mid_S$. The latter lemma shows 

<dir class="lemma">
Be $S = LL^T$ a symmetric postive definite matrix and $w \sim \mathcal{N}(0, S^{-1})$, then

$$ v = \frac{w}{ \mid  \mid w \mid  \mid_S} \text{ is uniformly distributed on } \{v \in \mathbb{R}^n:  \mid  \mid v \mid  \mid_S = 1 \} $$

</dir>
[proof]
Be $S = LL^T$ p.d  then  also $S^{-1} = L^{-1}L^{-T}$ is p.d with a square root. Be $\epsilon \sim \mathcal{N}(0, I)$ then $w \sim L^{-1}\epsilon$. Thus we can see that <br>
$$  \mid \mid w \mid  \mid_S^2 = w^T S w = w^T L^T L w = \epsilon^T L^{-T} L^T L L^{-1} \epsilon = \epsilon^T \epsilon =  \mid \mid \epsilon \mid  \mid_2^2$$ <br>
As a result we have that <br>
$$ v = \frac{w}{ \mid  \mid w \mid  \mid_S} = L^{-1} \frac{\epsilon}{||\epsilon||_2}$$ <br>
We know that the latter is uniformly distributed on the unit sphere. The linear transformation of a uniform distribution is again uniform on the image of the domain. Thus $v$ is uniformly distributed on the ellipse.
[/proof]

The covariance matrix of this distribution is hence given by

$$ \Sigma_{p(v)}[v] = \mathbb{E}_{p(v)}[vv^T] = \mathbb{E}_{p(w)}\left[\left( \frac{w}{ \mid  \mid w \mid  \mid_S}\right) \left( \frac{w}{ \mid  \mid w \mid  \mid_S}\right)^T\right]$$

$$  =\mathbb{E}_{p(w)}\left[ \frac{ww^T}{w^T S w} \right]= L^{-1}\mathbb{E}_{\mathcal{N}(\epsilon;0,I)}\left[ \frac{\epsilon \epsilon^T}{\epsilon^T \epsilon} \right] L^{-T}$$

This inner expectation looks nasty, but it really is not:
<dir class="lemma">
Be $x \sim \mathcal{N}(0, I)$ with $x \in \mathbb{R}^d$ then

$$ \mathbb{E}_{x} \left[\frac{xx^T}{x^Tx}\right] = \frac{1}{d} I $$

</dir>
[proof]
Be $X =  \frac{xx^T}{x^Tx}$ then we can deduct that $$X_{ii} = \frac{x_i^2}{\sum_{i=1}^dx_i^2}$$ and $$X_{ij} = \frac{x_ix_j}{\sum_{i=1}^dx_i^2}$$. <br>
Let's first try to get the diagonal elements. We have that<br>
$$ 1 = \mathbb{E}_x\left[ 1 \right] = \mathbb{E}_x\left[ \frac{\sum_{i=1}^d x_i}{\sum_{i=1}^d x_i} \right] = \sum_{i=1}^d \mathbb{E}_x\left[ \frac{ x_i}{\sum_{i=1}^d x_i} \right] = d \mathbb{E}_x\left[ \frac{ x_1}{\sum_{i=1}^d x_i} \right]$$ <br>
The last steps follows through the linearity of expectation and that all $x_i$ are iid. Hence $$\mathbb{E}_x[X_{ii}] = \frac{1}{d}$$. <br>
To show that the off-diagonal elements are zero, we can use that fact that both $X_{ij}$ and $-X_{ij}$ are equally distributed. Thus $$\mathbb{E}_x[X_{ij}] = -\mathbb{E}_x[X_{ij}]$$ and hence $$\mathbb{E}_x[X_{ij}] = 0$$
[/proof]

For the covariance matrix of the directional derivatives we hence have that

$$ \Sigma_{p(v)}[v] = \frac{1}{d} S^{-1} \qquad \Rightarrow \qquad \Sigma_{p(v)}[f_v'(x)] = \frac{1}{d} J_f(x)S^{-1}J_f(x)^T $$


## Conclusion

In summary, this post introduced the notion of a "local Lipschitz" property of a function. We generalized the one-dimensional notion to multivariate functions and showed that it can be quantified using operator norms of the Jacobian matrix. We also discussed how different norms can induce different notions of Lipschitzness and how this can be used to study the variation of functions in different ways. 
