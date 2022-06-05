---
layout: article
title: A tale of Fischer information in ML
key: A4
tags: Statistics, Math
aside:
    toc: true
comment: true
---

Fischer information plays a pivotal role in machine learning, as we will see in some way or another it will pop up in both Frequentist or Bayesian statistical paradigms. In this post, I will introduce the Fisher informations together with some important properties. We will go through several applications and relationships of the Fischer information to other important quantities.

## Notation and Definition

Suppose we have a parameteric statistical model $q_{\phi}(x)$ with parameter vector $\phi$ modeling some distribution. Our goal is to learn an unknown distributin $p(x)$ from which we have i.i.d. samples $x_1, \dots, x_N \sim p(x)$. In frequentist statistics the by far most common approach to learn $\phi$ is by maximizing the likelihood $\prod_{i=1}^N q_\phi(x_i)$ with respect to the parameter $\phi$. To assess the goodness of fit we can use the so called (Fischer) score, which we define as

$$ s_x(\phi) = \nabla_\phi \log q_\phi(x).$$

To justify this as measure of goodness, consider following claim:

[theorem]
Be $\log q_\phi(x)$ differentiable in $\phi$ almost everywhere and let the support of $q_\phi$ be independent of $\phi$. Then it holds that
$$ \mathbb{E}_{q_\phi(x)} \left[s_x(\phi)\right] =  \mathbb{E}_{q_\phi(x)}  \left[ \nabla_\phi \log q_\phi(x)\right] = 0 $$
[/theorem]
[proof]
As we can interchange integration and differentiation under above regularity conditions it holds that
$$ \begin{aligned}\mathbb{E}_{q_\phi(x)}  \left[ \nabla_\phi \log q_\phi(x)\right]  &= \int \frac{\nabla_\phi q_\phi(x)}{q_\phi(x)} q_\phi(x)dx \\ &= \int \nabla_\phi q (x)dx \\&= \nabla_\phi \int q_\phi(x)dx \\&= \nabla_\phi 1\\& = 0 \end{aligned} $$
[/proof]

Thus assuming we found some nice estimator $\phi^*$, we would expect that if it is a good fit we have that

$$ \mathbb{E}_{p(x)}\left[ s_{x}(\phi)\right] \approx \frac{1}{N} \sum_{i=1}^N \nabla_{\phi^*} \log q_{\phi^*}(x_i) \approx 0$$

So that's great, as $N \rightarrow \infty$ this will be zero if and only if $q_{\phi^*} = p$! Yet, anyway we are more interested in the deviation from zero to evaluate the goodness of fit. Thus let's look at the covariance of the score under the model. This idea leads us to the main topic of this post: The Fisher information (matirx)

<dir class="definition">
<b>(Fisher information):</b> Assuming the score $s(\phi)$ has a bounded second moment under $q_\phi$, then the the Fisher information (matrix) is defined as
$$F(\phi) := \Sigma_{q} \left(  s_x(\phi) \right) = \mathbb{E}_{q_\phi(x)} \left[ s_x(\phi)s_x(\phi)^T \right]$$
</dir>

We can again approximate the expectation the empirical data distribution, yielding the empirical Fisher $\hat{F}(\phi)$, which we can again evaluate at our selected parameter $\phi$

$$ \hat{F}(\phi) = \frac{1}{N} \sum_{i=1}^N s_{x_i}(\phi) s_{x_i}(\phi)^T $$

Note, that this estimate still measures the deviation of the score from zero, not from the actual mean (the mean is only zero if $p = q_\phi$! Hence this estimate is only unbiased if $x_i \sim q_\phi$. In any case, if a random variable has high Fisher information it implies that the absolute value of the score is often large.

A quick look at [wikipedia](https://en.wikipedia.org/wiki/Fisher_information#Regularity_conditions) gives us the following intuitive definition
<center> "In mathematical statistics, the Fisher information is a way of measuring the amount of information that an observable random variable $X$ carries about an unknown parameter $\phi$ of a distribution that models X."  </center>

So let's go through some examples 
### Example 1: Bernoulli model

Let's consider a coin flip experiment. The coin $X$ will be distributed according to a Bernoulli distribution 

$$Ber(x;\phi) = \phi^X (1-\phi)^{1-X}$$

Given $N$ observations $x_1, \dots, x_N \sim Ber(x;\phi)$, we can denote the loglikelihood as

$$ \log p(X|\phi) = \sum_{i=1}^N \log p(x_i|\phi) = (\sum_{i=1}^N x_i)\log \phi + (N - \sum_{i=1}^N x_i) \log (1-\phi) $$

Thus let's first derive the score function of this problem
$$ s_X(\phi) = \nabla_\phi  \log p(X|\phi) = \frac{(\sum_{i=1}^N x_i)}{\phi} - \frac{(N - \sum_{i=1}^N x_i)}{1-\phi} = \frac{\sum_{i=1}^N x_i (1-\phi) - N\phi + \sum_{i=1}^N x_i \phi}{\phi(1-\phi)} = \frac{\sum_{i=1}^N x_i  - N\phi}{\phi (1-\phi)}$$

Notice that the score is zero if the sum of Bernoulli trials equals the expecated on i.e. $N\phi$, a consequence of Theorem 1. Furthermore the magnitude of the score is determined by $1/(\phi (1-\phi))$ which is minimized if $\phi=0.5$ and approaches infinity for $\phi \rightarrow 1$ and $\phi \rightarrow 0$. As a result changes within the parameter near zero or one, will lead to a large score! Recall that the score equals the gradient of the log likelihood. A large score thus indicates that small changes within the parameter can strongly change the log likelihood.

So lets compute the corresponding Fisher information, as we defined it before

$$F(\phi) = \mathbb{E}_{q_\phi(x)} \left[ s(\phi)^2) \right] = \frac{\mathbb{E}_{q_\phi(x)} \left[ (\sum_{i=1}^N x_i  - N\phi)^2 \right] }{\phi^2 (1-\phi)^2}  = \frac{Var_{q_\phi(x)} \left( \sum_{i=1}^N x_i\right)  }{\phi^2 (1-\phi)^2} = \frac{N \phi (1-\phi)}{\phi^2 (1-\phi)^2} = \frac{N}{\phi (1-\phi)}$$

So that's cool, we can detect several properties of it. First of all the Fisher infomration increases linearly with the number of datapoints, we consider. Make's sense with more i.i.d. samples we learn more about the unknown parameter and each sample should contain the same amount of information. As we will later see, this will also be a universal property. Further we see that it is inversly proportianl to the variance of the distribution. This also makes sense that we can learn more about the unknown parameter from random variables with small variance, as we will see this property is to some degree also generalizable (but it is not that obvious as one would think intuitvely!). 

PLOT IT!

### Example 2: Exponential family

Let's look also at a more general family, which includes the previous example: The exponential family. Be $T(x)$ the sufficient statistics, $\eta$ the natural parameters and $Z,h$ the partition function and the base measure. Then an exponential family model is defined as 

$$ p_\eta (x) = h(x) \exp(\eta^T T(x) - Z(\eta)) $$

with $Z(\eta) = \log \int h(x) \exp(\eta^TT(x)) dx$. The log likelihood is thus for $N$ observations $x_1, ..., x_N$ is thus given by

$$ \log p_\eta(X) = \sum_{i=1}^N \eta^T T(x_i) - Z(\eta) + \log h(x_i)$$

For notational simplicity lets assume $N=1$, then the score of an exponential family is given by

$$ s_x(\eta) = \nabla_\eta \log p_\eta(x) = T(x) - \nabla_\eta Z(eta)$$

Notice that by this and Theorem 1 is follows that $\mathbb{E}_{p_\eta}[T(x)] = \nabla_\eta Z(eta)$ a nice property of exponential families. Let's get to the Fisher information 

$$ F(\eta) = \mathbb{E}_{p_\eta} \left[ (T(x) - \nabla_\eta Z(eta))(T(x) - \nabla_\eta Z(eta))^T   \right] = \Sigma_{p_\eta(x_i)}(T(x))$$

Thus the Fisher information equals the covariance of the sufficient statistic $T(x)$! But wait the Bernoulli distribution is an exponential family, right? And we just calculated that in this case the Fisher information equals the inverse variance. So isn't this a contradiction?

No, we just discovered that Fisher information can change upon a change in parameterization. Remember, before the paramter $\phi$ was the probability of success. In contrast the natural paramter for a Bernoulli distribution is given by $\eta = \log \frac{\phi}{1 - \phi}$. But luckily as we will later see, translate between different parameterization rather easily.

## Important properties of the Fisher information

If we observe i.i.d. samples, each sample should contain the same amount of information. 

<dir class="lemma">
Be $X$ and $Y$ be two independent random variables, then

$$ F_{X,Y}(\phi) = F_{X}(\phi) + F_{Y}(\phi)$$
</dir>
[proof]
We can write
$$ \begin{aligned}
    F_{X,Y}(\phi) &= \mathbb{E}_{X,Y}\left[ s_{X,Y}(\phi)s_{X,Y}(\phi)^T \right]\\ 
    & = \int \int  \nabla_\phi \log p(x,y|\phi) \left( \nabla_\phi \log p(x,y|\phi) \right)^T p(x)p(y)dxdy\\
    &= \int \int  \nabla_\phi (\log p(x|\phi) + \log p(y|\phi)) \left( \nabla_\phi (\log p(x|\phi) + \log p(y|\phi)) \right)^T p(x)p(y)dxdy\\ 
    &= \int \nabla_\phi \log p(x|\phi)(\nabla_\phi \log p(x|\phi))^Tp(x) dx + \int \nabla_\phi \log p(y|\phi)(\nabla_\phi \log p(y|\phi))^Tp(y) dx\\ 
    & \quad +  \int \int \nabla_\phi \log p(x|\phi) (\nabla_\phi \log p(y|\phi))^T p(x)p(y)dxdy\\ 
    & \quad +  \int \int \nabla_\phi \log p(y|\phi) (\nabla_\phi \log p(x|\phi))^T p(y)p(x)dydx\\ 
    &= \mathbb{E}_{X}[s_X(\phi) s_X(\phi)^T] + \mathbb{E}_{Y}[s_Y(\phi) s_Y(\phi)^T] + \int  \underbrace{\mathbb{E}_X[ s_{X}(\phi)]}_{= 0} s_Y(\phi)^T p(y)dy + \int  \underbrace{\mathbb{E}_Y[ s_{Y}(\phi)]}_{= 0} s_X(\phi)^T p(x)dx\\ 
    & = F_X(\phi) + F_Y(\phi) 
\end{aligned}$$
[/proof]

So this is great and by it we can easily explain how the Fisher information behaves under the i.i.d. assumpiton:

<dir class="theorem">
If $X_1, \dots, X_N$ are i.i.d. then

$$ F_{X_1, \dots, X_n}(\phi) = N F_{X_1}(\phi)$$
</dir>

We can extend Lemma 1 to a more general case, were $X$ and $Y$ are not independent:

<dir class="lemma">
Be $X$ and $Y$ two random variables, jointly distributed according to $p(X,Y)$. Then 

$$ F_{X,Y}(\phi) = F_{X|Y}(\phi) + F_Y(\phi) = F_{Y|X}(\phi) + F_X(\phi)$$

where we define the conditional Fisher information as

$$ F_{X|Y} = \mathbb{E}_Y\left[ F_{X|y = Y}(\phi) \right] = \mathbb{E}_Y\left[ \int \nabla_\phi \log p(x|y = Y) (\nabla_\phi \log p(x|y = Y))^T p(x|y=Y)  \right]$$
</dir>
[proof]
As above just using the product rule $\log p(X,Y) = \log p(X|Y) + \log p(Y) = \log p(Y|X) + \log p(X)$
[/proof]

We already saw that the Fisher information depends on the paramteriztion of our statistical model. So let's try to exactly quantify how different parameterization do change the Fisher information.

<dir class="lemma">
Be f a totally differentaibe map from parameter with $\eta = f(\phi)$, where both parameters encode the same statistical model. Be $J_f = \nabla_\phi f(\phi)$ the Jacobian matrix, then
$$ F_{x}(\phi) = J_f^T F_X(f(\phi)) J_f $$
</dir>
[proof]
This simply folows from the chain rule of differentiation. Notice that we can write
$$ \nabla_\phi \log q_{f(\phi)}(x) = \nabla_\phi f(\phi) \nabla_\eta \log q_\eta(x) = J_f \nabla_\eta \log q_\eta(x)$$
By the definition of Fisher information we have that

[/proof]

Recall that the natural parameters of a Bernoulli distribution are $\eta = f(\phi) = \log \frac{\phi}{1-\phi}$. Thus $\nabla_\phi f(\phi) = \frac{1}{\phi(1-\phi)}$, we had that the Fisher information of an exponential family equals the covariance of the test statistic. For a Bernoulli variable we thus have
$$  F_x(\eta) = f^{-1}(\eta) (1- f^{-1}/\eta) = \phi (1-\phi)$$
Thus by applying the above lemma we get
$$ F_x(\phi) = \frac{\phi (1-\phi)}{(\phi (1-\phi)^2} = \frac{1}{\phi (1-\phi)}$$
which is exactly what we obtained by explicitly calculating it.

Within the Bernoulli example we already observed an close relationship to the variance of the random variable. This relationship is not as universal as you would guess! 

NOT SURE IF THERE ACTUALLY IS ONE ...

Another very usefull property is the connection to the hessian of the log likelihood. 

> Claim: Be $\log q_\phi(x)$ is twice differentiable and be $H_{\log q (x)}$ the corresponding Hessian. Under certain regularity conditions it holds that
> $$ F_X(\phi) = -\mathbb{E}_{q (x)} \left[ H_{\log q (x)}  \right]$$
> Proof: We can write the Hessian of the log likelihood as follows
> $$ H_{\log q (x)}= \nabla_\phi \nabla_\phi^T \log q (x) = \nabla_\phi \frac{\nabla_\phi^T q_\phi(x)}{q (x)} = \frac{ \nabla_\phi \nabla_\phi^T q (x) q_\phi(x) - \nabla_\phi q (x) \nabla_\phi^T q (x)}{q_\phi(x)^2} = \frac{H_{ q (x)}}{q (x)} - \frac{\nabla_\phi q (x) \nabla_\phi^T q (x)}{q (x)^2} $$
> Which simply follows from the chain and quotient rule of differentiation. Let's apply the expectation to the first term.
> $$ \mathbb{E}_{q (x)} \left[  \frac{H_{ q (x)}}{q (x)}  \right] = \int H_{ q (x)} dx = \int \nabla_\phi \nabla_\phi^T q (x)dx = \nabla_\phi \nabla_\phi^T \int q (x) dx = \nabla_\phi \nabla_\phi^T 1 = 0$$
> So this term fanishes, let's apply it to the second term
> $$ \mathbb{E}_{q (x)} \left[ - \frac{\nabla_\phi q (x) \nabla_\phi^T q (x)}{q (x)^2} \right] = - \mathbb{E}_{q (x)} \left[ \left( \frac{\nabla_\phi q (x)}{q (x)}\right) \left( \frac{\nabla_\phi q (x)}{q (x)}\right)^T \right] = - \mathbb{E}_{q (x)} \left[ \left( \nabla_\phi \log q (x)\right) \left( \nabla_\phi \log q (x)\right)^T \right]$$
> which proves the statement.

This interpretations also gives use a bunch of additional properties, 

> Claim: The Fisher information satisfies the following properties
>  $F(\phi)$ is symmetric and positive semi-definite.
>  $F(\phi)$ is positive definite if the statistical model is identifiable (there cannot exist two parameters $\phi_i \neq \phi_j$ such that $p_{\phi_i} = p_{\phi_j}$)
>  If $[\phi]_i$ and $[\phi]_j$ are independent then $[F(\phi)]_{ij} = 0$ i.e. then $[\phi]_i$ and $[\phi]_j$ can be estimated independently of each other.
>  Be $\eta = f(\phi)$ an alternative reparameterization, then $F(\eta) = J_f^T F(\phi) J_f$, where $J_f$ denotes the Jacobian matrix of $f$.
> 
> Proof:: You can do this, or I some day in the future ...



## Some applications in frequentist statistics...

Already within the introduction we closely related the Fisher Information to maximum likelihood estimation. Recall that in frequentist statistic we typically have to do the following:

 Propose an estimator (typically a point estiamte) of the parameter.
 Test wheather it's value aligns with the data.
 Derive confidence intervals.

As we will see in each of this steps the Fisher information will be involved in some way or another.

### Proposing an estimator

An estimator is typically just a function $f : X^N \rightarrow \phi$, which maps a set of observations to a specific parameter. Yet in certain cases it may be less practical to work with the raw data and we instead want to preprocess it. We may even use a statistic to summarize the data, i.e. $T: X^N \rightarrow \mathcal{T}$. This may does simplify the design of an estiamtor significantly as we now only have to consider a single or a few statistics. Yet, do we actually use information by just considering the statistic? The most information should always be within the raw data, right? In fact this is indeed true (according to Fisher information) !

> Claim: (Data-processing inequality) Be $X$ an random variable and be $T(X)$ an arbitrary transformation independent of $\phi$. Then
> $$ F_{T(X)}(\phi) \leq F_X(\phi)$$
> Where for matrices we use the standard Loewner order i.e. $A \leq B \iff A - B$ is postive semi definite. The inequality is a equality if and only if $T$ is a sufficient statistic.
> 
> Proof: As the Fisher information satisfies the additive chain rule it follows that for any random variable $Y$ we have that
> $$ F_{Y, X}(\phi) = F_{Y| X}(\phi) + F_{X}(\phi) \geq F_X(\phi)$$
> where the inequality holds because $F_{Y| X}(\phi)$ is postive semi definite.
> Thus if follows that
> $$ F_{T(X)}(\phi) \leq F_{T(X), X}(\phi) = \underbrace{F_{T(X)|X}(\phi)}_{=0} + F_X(\phi) = F_X(\phi).$$
> Here $F_{T(X)|X}(\phi) = 0$ as $T(X)$ is a deterministic transform of $X$ and thus independent of $\phi$ given $X$.

So that's nice that Fisher informations does indeed follow our intuition. So let's start to actually do inference. The most common type to obtain an estimator $f$ of data is the maximum likelihood method. We typically propse some family of parameteric models $\mathcal{F} =\{ q | \phi \in \phi \}$, given some data $x_1, ..., x_n \sim p$ we then select 
$$ \phi = \argmax_{\phi \in \phi} \sum_{i=1}^n \log q_\phi(x_i)$$

### TODO MLE IS ASYMPTOTICALLY NORMAL

### TODO APPROXIMATE CONFIDENCE INTERVALS USING FISHER INFO

### TODO EXPERIMENTAL DESIGN ACCORDING TO FISHER INFO


## TODO Bayesian use of Fisher info!

### TODO BERNSTEIN VON MIES THEOREM

### TODO JEFREY PRIORS AND SO ONE...









## References
<a id="1">[1]</a> 
Alexander Ly, Maarten Marsman, Josine Verhagen, Raoul
Grasman and Eric-Jan Wagenmakers (2017). 
A Tutorial on Fisher Information 

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1037.2696&rep=rep1&type=pdf

https://agustinus.kristia.de/techblog/2018/03/11/fisher-information/

https://awni.github.io/intro-fisher-information/

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.323.8983&rep=rep1&type=pdf
