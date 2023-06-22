---
layout: article
title: Kernel mean embeddings
tags: Math, Statistics, ML
cover: https://www.mdpi.com/pharmaceutics/pharmaceutics-12-00271/article_deploy/html/images/pharmaceutics-12-00271-g005.png
comment: true
---

# Kernel mean embeddings

Working with distributions is fundamental in both machine learning and statistics. Some methods may only require the existence of an underlying true distribution from which we learn point etimates, yet others are fundally based on probability distributions. Unfortunatly working with distribution is hard. Often we just have data points, so we first have to estimate the underling distribution or density. In other cases we may want to compute "distances" or "similarities" between certain distributions, which also is hard as it requires to compare two functions. In general: **Working with distributions is hard**.

In general it is much easier to work in a vector space, or more general in *Hilbert spaces* i.e. a vector space with an inner product. Here we can easily evaluate distances with norms , we can evaluate the angle between vectors with inner prodcuts and can estimate a function easily from data points using regression algorithms.

This motivates to **embbed distributions in Hilbert spaces**. The basic idea is to map distributions in **Reproducing Kernel Hilbert spaces (RKHS)**, allowing one to compare and manipulate distributions using Hilbert space operations such as inner products, distances, projections, linear transformations, and more. It can be seen as the generalization of the *kernel trick* from individual data points to arbitrary distributions. While it is helpfull to already know some of the standard *kernel methods* like Support Vector Machines or Kernel Least squares regression, it is not required. I will try to motivate it from first principles.

## Embedding distributions as vectors

Let's start with a simple motivation. Our problem involves some complex objects i.e. "distributions" and somehow we want to embedd this into standard euclidean space i.e. we want to represent it as a vector.

If you think about it we have the same problem in Machine Learing i.e. we often have complex data e.g. text and want to transform it into a vector representation i.e. we want to find a **feature mapping** to a vector e.g. word2vec. We then can apply standard ML techniques to e.g. anylze the content of documents. It is important that these features describe our complex objects as best as possible otherwise the learning algorithm will perform pourly on the task.

Let's say we have two real random variables $X \sim p$ and $Y \sim q$ following the distributions $p$ and $q$ respectively. We know a lot of point estimates for distributions, the argubly most common is the *mean* of a distribution. So for the start we may propose 
$$ \mu_X = \mathbb{E}_p[X] \quad \text{ and } \quad \mu_Y = \mathbb{E}_q[Y] $$
as feature map. Both $\mu_X, \mu_Y \in \mathbb{R}^d$ and we now easily can compute e.g. the distance between $p$ and $q$ as the distance $d(p,q) = ||\mu_X - \mu_Y||$. But is this a "usefull" representation? This partially depends on your goal, if the only quantity your are interested in is the mean then go ahead. Yet, many very different distributions can have the same mean, in other words this feature mapping is not *injective*. This is demonstrated in the below figure, which shows a collection of distributions. As one can see many fundamentally different distributions are embedded onto the same point in feature space...

% TODO FIGURE

To fix this, we may want to extend our feature mapping to map into an higher dimensional space. Another typically important quantity we are often interested in is the **variance** of the distribution. Lets consider a mapping $\phi : \mathbb{R}^d \rightarrow \mathbb{R}^n$ then e.g. lets consider

$$ \phi(X) = \begin{pmatrix} X \\ (X - \mathbb{E}[X])^2 \end{pmatrix} $$

Thus we can obtain our new embedding as

$$ \mu_X = \mathbb{E}[\phi(X)] \quad \text{and} \quad \mu_Y = \mathbb{E}[\phi(Y)]$$

Let's look at the same figure as above, but the new two dimensional embedding. As we can see we now atleast can "distinguish" most of the distributions also in feature space. In fact if we would just work with normal distributions, then this embedding would contain all necessary information. Yet, in general there are still many "different" distributions which can have the same mean and variance. For example in the below figure the red and green distributions looks similar but infact the green one is a Normal distribution where the red one is a StudentT distribution!

% TODO Add figure

So how we can make our feature mapping more sensitive to general distributions, at best that we can gurantee that $$||\mu_X - \mu_Y|| = 0 \iff X = Y$$ i.e. the mapping should be injective. Todo so recall that $$X = Y$$ is satisfied if all *moments* of the distributions are the same! So we can straight forwardly extend our approach by considering the feature mapping

$$ \phi_n(X) = \begin{pmatrix} X \\ X^2\\ \vdots \\ X^n \end{pmatrix} \quad \text{ and } \quad \mu_X = \mathbb{E}[\phi_n(X)] $$

As we increase $$n\rightarrow \infty$$ this should work, right? Well ... no, there are a few problems we have to care about:
* Not any random variable has an infinite number of finite moments i.e. the StudentT distributions only has finite moments up to the number of degrees of freedom.
* This is only injective if $$n = \infty$$ ... .

Alright, let's first get rid of the first problem. Recall that another way to uniquely characterize a distribution is it *cumulative distribution function* (CDF). In contrast to all the moments, this always exists! Be $x_1, \dots, x_n \in \mathbb{R}^d$ with $x_1 \neq x_n$ and consider the feature mapping 

$$ \phi_n(X) = \begin{pmatrix} I(x_1 \leq X) \\ I(x_2 \leq X)\\ \vdots \\ I(x_n \leq X) \end{pmatrix} \quad \text{ and } \quad \mu_X = \mathbb{E}[\phi_n(X)] $$

it is easy to see that each element converge to the CDF evaluate at $x_i$. We can say that this becomes injective if and only if we do this for all real numbers. In fact this infinite dimensional vector then just becomes the CDF itself and the vector space becomes a function space equiped with the usual inner product. So this theoretically works, but computing distance then amounts to evaluate $||\mu_X - \mu_Y|| = \int_\mathbb{R} ||F_X(x) - F_Y(x)|| dx$, which itself is intractable...

This shouldn't surprise you at all as distributions in general are infinite dimensional objects, so any *injective* mapping must also map to a infinite dimensional feature space. Yet, such infinite dimensional feature spaces are only usefull in practise if we know a simple way on how to evaluate an inner product on them and thus also e.g. distances. This motivites the use of the so-called *kernel trick*.

## Kernel trick - Reproducing Kernel Hilber Spaces (RKHS)

TODO

## Kernel methods for distributions

TODO



$$\begin{aligned} p(x_t | Y_{0:t}) &= p(x_t| Y_{0:t-1}, y_t)\quad \quad \ \text{Change in notation}\\& = \frac{p(x_t, y_t| Y_{0:t-1})}{p(y_t)} \quad \quad \text{Definition conditional density}\\& = \frac{p(y_t|Y_{0:t-1}, x_t)p(x_t|Y_{0:t-1})}{p(y_t)} \quad  \text{Product rule} \\  &=\frac{p(y_t|x_t)p(x_t|Y_{0:t-1})}{p(y_t)} \quad \text{Conditional independence y given x}  \end{aligned} $$





## References

- https://arxiv.org/pdf/1605.09522.pdf
- https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions