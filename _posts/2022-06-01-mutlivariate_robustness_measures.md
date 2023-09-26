---
layout: article
title: Local variation of functions
tags: Math
aside:
    toc: true
cover: https://upload.wikimedia.org/wikipedia/commons/5/58/Lipschitz_Visualisierung.gif
comment: true
---

## Operator norms
### Univariate case

In the one-dimensional case, i.e., $f: \mathbb{R} \rightarrow \mathbb{R}$, the local variation of a function is given by its derivative, i.e., the instantaneous rate of change. If the absolute slope is bounded by $L$ within a set $S \subset \mathcal{X}$, then we know that $ \mid f(x) - f(y) \mid \leq L \cdot \mid x-y \mid$ for all $x, y \in B_\epsilon(x)$. To see the reverse direction, consider some $x \in S$, then

$$ \mid f(x) - f(x + h) \mid \leq L*h \iff \frac{\mid f(x) - f(x+ h) \mid}{h} \leq L \iff \lim_{h\rightarrow 0} \frac{\mid f(x) - f(x+ h) \mid}{h} \leq L  \iff \mid f'(x) \mid \leq L $$

Conversely if $ \mid f'(x) \mid \leq L$ and be $x, y \in S$ then by the mean value theorem, there exists some $\xi \in (x, y)$ such that

$$ \mid f'(\xi) \mid = \frac{\mid f(x)-f(y) \mid}{\mid x-y \mid} \iff \mid f(x)-f(y) \mid = \mid f'(\xi) \mid \mid x-y \mid \iff \mid f(x)-f(y) \mid \leq L \mid x-y \mid $$

This is known as a *local Lipschitz* property. In this post, we will investigate this property in the case of a multivariate function. Further, we relate alternative approaches to quantifying the local variation of a function.

### Multivariate case

We consider a totally differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$. The first-order derivative is thus given by its Jacobian matrix $J_f(x) \in \mathbb{R}^{n \times m}$. Analogously, we define the *local Lipschitz* property as following

<div class="definition">
 A function $f$ and set $S \subset \mathcal{X}$ is called locally L-Lipschitz for $L \geq 0$ in the (vector) norm $ \mid  \mid  \cdot  \mid  \mid $ if for all $x, y \in S$

  $$  \mid  \mid f(x) - f(y) \mid  \mid  \leq L  \mid  \mid x - y \mid  \mid $$
</div>

Notice that the first-order derivative is now a matrix. Thus we require the notion of a matrix or operator norm. Given two vector norms $ \mid  \mid  \cdot  \mid  \mid_a$ and  $\mid  \mid \cdot \mid  \mid_b$, the operator norm of a linear map represented through a matrix $A$ is given by

$$  \mid  \mid  A  \mid  \mid = \sup_{ \mid  \mid  x  \mid  \mid _a = 1}  \mid  \mid  Ax \mid  \mid _b$$

The spectral norm for example is induced by the Euclidean vector norms and can be shown to be

$$  \mid  \mid A \mid  \mid _2 = \sqrt{ \lambda_{max}(A^T A)} =  \sigma_{max}(A) $$

, where $\sigma_{max}$ denotes the largest singular value of $A$.

This is great, but let's first try to understand why relating the Lipschitz property with the Jacobian operator norm is not straightforward.  First of all, recall that we required the mean value theorem in the 1d case. So let's first try to generalize this to the multivariate case.

<div class="lemma">
Be $f$ differentiable on an open set $S \subset \mathbb{R}^n$. Consider some $x, y \in S$, let $L_{xy}$ be the line segment connecting them. Then if $L_{xy} \subset S$ there exists an $\xi \in L_{xy}$ such that
$$ f(x) - f(y) = J_f(\xi)(x-y)$$ 
</div>
[proof]
The line segment is given by $c(t) = tx + (1-t)y$ for $t \in [0,1]$. Consider the one-dimensional function $g(t) = f(c(t))$, then by the standard mean value theorem there exists some $\tau$ such that $$ g(1) - g(0) = (1-0) g'(\tau) \iff f(x) -f(y) = J_f(c(\tau))(x-y)$$
[/proof]

This is straightforward but very restrictive as it only works on linear lines between points and can only be applied if this line is fully contained in $S$. We can guarantee these constraints by restricting $S$ to be convex. Recall a set is called convex if for all $x, y \in S$ and $\lambda \in [0,1]$ it holds that $\lambda x + (1-\lambda)y \in S$, exactly what we require for the proof to work. We now can state a more general result, which is known as the *multivariate mean value theorem*.

<div class="lemma">
Be $f$ differentiable on an open and convex set $S \subset \mathbb{R}^n$. Then for any $x, y \in S$ we have that
$$  \mid  \mid f(x) - f(y) \mid  \mid _a \leq  \mid  \mid J_f(\xi) \mid  \mid \cdot  \mid  \mid x-y \mid  \mid _b$$ 
</div>
[proof]
By Lemma 1 we know that for each fixed $x, y \in S$ there is an $\xi \in S$ such that
$$ f(x) - f(y) = J_f(\xi) (x-y) \iff  \mid  \mid  f(x) - f(y)  \mid  \mid _a =  \mid  \mid  J_f(\xi) (x-y)  \mid  \mid _a$$
By the fundamental inequality of operator norm
$$  \mid  \mid  J_f(\xi) (x-y)  \mid  \mid _a \leq  \mid  \mid  J_f(\xi) \mid  \mid   \mid  \mid  (x-y)  \mid  \mid _b$$
which proofs the statement.
[/proof]

This now allows us to state the local Lipschitz criteria.

<dir class="theorem">
 The function $f$ is locally L-Lipschitz on an open and convex set $S$ in norm $ \mid  \mid \cdot  \mid  \mid _a$ and $ \mid  \mid \cdot \mid  \mid _b$ if and only if 
 
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

That is nice and can help us to restrict the Lipschitz constant, e.g., for neural networks. A neural network can typically be written in a "layer-wise" composition of linear maps and nonlinear activation functions, i.e., for a single layer ReLU $g(x) = \max(x, 0)$ neural network 

$$ f(x) = g_1(W_1x + b)$$

By the chain rule, the Jacobian takes the following form

$$ J_f(x) = J_{g}(h_1)W_1  \text{ with } h_1 = Wx + b $$

where $J_{g}$ takes the form of a diagonal matrix which is either zero or one.

Given some convex set $x$, we can thus compute the local Lipschitz constant by solving the following optimization problem

$$ L = \sup_{x \in S} \mid \mid  J_f(x)\mid \mid  $$

This is generally hard because, as we saw, the Jacobian does depend on the hidden activations, hence would require some constrained numerical optimization.

Yet we can find more easily an upper bound by using the submultiplicative property of the operator norm.

$$ L = \sup_{x \in S} \mid \mid  J_f(x)\mid \mid  = \sup_{x \in S} \mid \mid  J_{ReLU}(h(x))W\mid \mid  \leq  \sup_{x \in S} \mid \mid  J_{ReLU}(h(x))\mid \mid  \cdot \mid \mid W\mid \mid  \leq \mid \mid W\mid \mid  $$

Hence we can upper bound the neural net Lipschitz constant by constraining the operator norm on the weights. This is the foundation of how Lipschitz-Networks are constructed or why spectral normalization works (it bounds the L2 operator norm).

## Probabilistic interpretation

As one can guess, the above statements can be interpreted as "worst-case" guarantees. This simply follows by definition of operator norms: It equals the worst-case absolute slope in any possible direction. Yet, sometimes we may be interested in average-case guarantees.

Be $p(v)$ a uniform distribution on a unit-sphere induced by some norm $\mid \mid \cdot \mid \mid _a$. For example, in the case of the L2 norm, we have that $v \sim p(v) \iff v = \frac{w}{\mid \mid w\mid \mid _2}$ with $w \sim \mathcal{N}(0, I)$. We already encountered the notion of directional derivatives in the previous theorem. Bounding the directional derivatives will also bound the local Lipschitz constant. Thus let's investigate its distribution!

A prior $p(v)$ on a unit-sphere induces a distribution on the directional derivatives as $f_v'(x) = J_f(x)v$. This distribution will have interesting properties. Let's again consider the L2 norm and compute the first and second moments. Notice that given a uniform distribution on the L2 unit sphere, we have that $$\mathbb{E}_{p(v)}[v] = 0$$ and $$\Sigma_{p(v)}[v] = \frac{1}{d}I$$ thus

$$ \mathbb{E}_{p(v)}[f_v'(x)] = J_f(x)\mathbb{E}_{p(v)}[v] = 0$$

$$ \Sigma_{p(v)}[f_v'(x)] = \mathbb{E}_{p(v)}[ J_f(x)v(v)^T] = \frac{1}{d} J_f(x)J_f(x)^T $$

So this is a nice closed-form solution and again involves the Jacobian matrix. Recall these results are induced by the L2 norm, so they also should be related to the operator norm induced by the L2 norm. Recall that $\mid \mid  J_f(x)\mid \mid $ equals the largest singular value and thus is related to the eigenvalue of $J_f(x)^T J_f(x)$. Be $v$ the corresponding eigenvector then

$$ J_f(x)^T J_f(x) = \lambda v \iff J_f(x)J_f(x)^T(J_f(x)v) = \lambda (J_f(x)v)$$

and hence both share the same eigenvalues. As $J_f(x)J_f(x)^T$ is up to a constant the covariance matrix of directional derivatives, the eigenvector with the largest eigenvalue hence points in the direction of the largest variation as required by the definition of the operator norm.

Yet for example if we instead want to control the average variation of the function, this suggests controlling all the eigenvalues i.e. by controlling the trace of the covariance.

## Non-standard norms

Both the operator norm as well as the covariance matrix of the directional derivatives can be hard to compute for non-standard norms. We will investigate here the problem in the case of a more general "euclidean-like" norm, the *elliptical norm*.

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

$$ p(z|x) = p(z,x)/p(x)  $$


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
Be $S$ a symmetric postive definite matrix and $w \sim \mathcal{N}(0, S^{-1})$, then

$$ v = \frac{w}{ \mid  \mid w \mid  \mid_S} \text{ is uniformly distributed on } \{v \in \mathbb{R}^n:  \mid  \mid v \mid  \mid_S = 1 \} $$

</dir>
[proof]
Because $S$ is p.d also $S^{-1}$ is and thus there exists a square root $S^{-1} = L^{-1}L^{-T}$. Be $\epsilon \sim \mathcal{N}(0, I)$ then $w \sim L^{-1}\epsilon$. We thus can write <br>
$$  \mid \mid w \mid  \mid_S^2 = w^T S w = w^T L^T L w = \epsilon^T L^{-T} L^T L L^{-1} \epsilon = \epsilon^T \epsilon =  \mid \mid \epsilon \mid  \mid_2^2$$ <br>
as $S$ is symmetric. As $\epsilon$ is a unit-normal vector, this proves the statement.
[/proof]

So now the question is how to modify $S$ such that $w$ becomes a unit normal distribution. And indeed, we can always modify it as following

$$ w \sim \mathcal{N}(0, S^{-1}) \iff w = L \epsilon \iff \epsilon = L^{-1}w $$

Plugging this back we have that

$$ \mid \mid v \mid  \mid_S^2 = \mid \mid w \mid  \mid_{S^{-1}}^2 = \mid \mid L^{-1}w \mid  \mid_{I}^2 = \mid \mid L^{-1}w \mid  \mid_{2}^2 $$

Now we have some interesting statement about $v$. We can easily compute its covariance as following

$$ \Sigma_{p(v)}[v] = \mathbb{E}_{p(v)}[vv^T] = \mathbb{E}_{p(v)}\left[\left( \frac{w}{ \mid  \mid w \mid  \mid_S}\right) \left( \frac{w}{ \mid  \mid w \mid  \mid_S}\right)^T\right]$$

$$ = \frac{1}{ \mid  \mid w \mid  \mid_S^2}\mathbb{E}_{p(v)}\left[ww^T\right] = \frac{S}{ \mid  \mid w \mid  \mid_S^2}$$

The denominator does not depend on $S$ and therefore all the eigenvalues of $\Sigma_{p(v)}[v]$ are proportional to the eigenvalues of $S$. By choosing $S = I$ we obtain the standard covariance matrix, but by choosing $S$ with the inverse eigenvalues of $J_f(x)J_f(x)^T$ we can change the covariance of $v$.

## Conclusion

In summary, this post introduced the notion of a "local Lipschitz" property of a function. We generalized the one-dimensional notion to multivariate functions and showed that it can be quantified using operator norms of the Jacobian matrix. We also discussed how different norms can induce different notions of Lipschitzness and how this can be used to study the variation of functions in different ways. 

## References
* http://www.math.toronto.edu/courses/mat237y1/20199/notes/Chapter2/S2.4.html#:~:text=The%20Mean%20Value%20theorem%20of,(a)b%E2%88%92a.