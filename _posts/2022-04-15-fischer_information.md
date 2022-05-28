---
layout: article
title: The Fischer information and why you should know it
key: A4
tags: Statistics, Math
comment: true
---

Fischer information plays a pivotal role in machine learning, as we will see in some way or another it will pop up in both Frequentist or Bayesian statistical paradigms. As we will also discuss several applications and relationships of the Fischer information to other important quantities.

## Notation and Definition

Suppose we have a parameteric statistical model $p_{\theta}(x)$ with parameter vector $\theta$ modeling some distribution. Our goal is to learn an unknown distributin $p^*(x)$ from which we have i.i.d. samples $x_1, \dots, x_N \sim p^*(x)$. In frequentist statistics the by far most common approach to learn $\theta$ is by maximizing the likelihood $\prod_{i=1}^N p_{\theta}(x_i)$ with respect to the parameter $\theta$. To assess the goodness of fit we can use the so called **(Fischer) score**, which we define as
$$ s(\theta) = \nabla_\theta \log p_\theta(x).$$
To justify this as measure of goodness, consider following claim:

> **Claim**: Be $\log p_\theta(x)$ continously differentiable in both $x$ and $\theta$. Then it holds that
> $$ \mathbb{E}_{p_\theta(x)} \left[s(\theta)\right] =  \mathbb{E}_{p_\theta(x)}  \left[ \nabla_\theta \log p_\theta(x)\right] = 0 $$
> **Proof**: As $\log p_\theta(x)$ is continously differentible in both arguments, we can interchange integration and differentiation, thus one can easily see that
> $$ \begin{aligned}\mathbb{E}_{p_\theta(x)}  \left[ \nabla_\theta \log p_\theta(x)\right]  &= \int \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} p_\theta(x)dx \\ &= \int \nabla_\theta p_\theta (x)dx \\&= \nabla_\theta \int p_\theta(x)dx \\&= \nabla_\theta 1\\& = 0 \end{aligned} $$
Thus assuming we found some nice estimator $\theta^*$, we would expect that if it is a good fit we have that
$$ \mathbb{E}_{p^*(x)}\left[ s(\theta^*)\right] \approx \frac{1}{N} \sum_{i=1}^N \nabla_{\theta^*} \log p_{\theta^*}(x_i) \approx 0$$

So that's great, as $N \rightarrow \infty$ this will be zero if and only if $p_{\theta^*} = p^*$! Yet, we may are also interested in how certain we are in our estiamte. Thus let's look at the variance

$$ Var_{p_\theta(x)} \left(  s(\theta) \right) = \mathbb{E}_{p_\theta(x)} \left[ (s(\theta) - 0)(s(\theta) - 0)^T \right] = \mathbb{E}_{p_\theta(x)} \left[ s(\theta)s(\theta)^T \right] := F(\theta).$$

We call $F$ the **Fisher Information Matrix**. We can again approximate the expectation in $F$ using our data, yielding the **Empirical Fisher**, which we can again evaluate the fit of our selected parameter $\theta^*$
$$ \hat{F}(\theta) = \frac{1}{N} \sum_{i=1}^N s(\theta^*) s(\theta^*)^T $$
Note that this estimate still measures the deviation from 0, not from the actual mean (the mean is only zero if $p^* = p_{\theta^*}$, so it is the deviation assuming our parameter choice is optimal). 

If the random variable $X$ is univariate, then $F(\theta)$ will be a scalar, which is now typically called the **Fisher Information**. So let's try to build some intuition about this measure, a quick glance on Wikipedia will reveal

> **Fisher information**: "In mathematical statistics, the Fisher information is a way of measuring the amount of information that an observable random variable $X$ carries about an unknown parameter $\theta$ of a distribution that models X."

So let's test this hypothesis on a few easy examples. 

### Bernoulli Example

Lets consider a coin flip experiment. The coin $X$ will be distributed according to a Bernoulli distribution 
$$Ber(x;\theta) = \theta^X (1-\theta)^{1-X}$$
Given $N$ observations $x_1, \dots, x_N \sim Ber(x;\theta)$, we can denote the loglikelihood as
$$ \log p(X|\theta) = \sum_{i=1}^N \log p(x_i|\theta) = (\sum_{i=1}^N x_i)\log \theta + (N - \sum_{i=1}^N x_i) \log (1-\theta) $$
Thus let's first derive the score function of this problem
$$ s(\theta) = \nabla_\theta  \log p(X|\theta) = \frac{(\sum_{i=1}^N x_i)}{\theta} - \frac{(N - \sum_{i=1}^N x_i)}{1-\theta} = \frac{\sum_{i=1}^N x_i (1-\theta) - N\theta + \sum_{i=1}^N x_i \theta}{\theta(1-\theta)} = \frac{\sum_{i=1}^N x_i  - N\theta}{\theta (1-\theta)} = -F(\theta)$$

Notice that the score is zero if the sum of Bernoulli trials equals the expecated on i.e. $N\theta$. Furthermore the magnitude of the score is determined by $\theta (1-\theta)$ which is maximized if $\theta=0.5$ and approaches zero for $\theta \rightarrow 1$ and $\theta \rightarrow 0$. As a result changes within the parameter near zero or one, will lead to a large score! Recall that the score equals the gradient of the log likelihood. A large score thus indicates that small changes within the parameter can strongly change the log likelihood. 


So lets compute the corresponding Fisher information. NOTE: This computation will rely on our current definitions, we will later see that there is a much simpler way...

$$F(\theta) = \mathbb{E}_{p_\theta(x)} \left[ s(\theta)^2) \right] = \mathbb{E}_{p_\theta(x)} \left[ \frac{(\sum_{i=1}^N x_i  - N\theta)^2}{\theta^2 (1-\theta)^2} \right] = \mathbb{E}_{p_\theta(x)} \left[ s(\theta)^2) \right] = \frac{\mathbb{E}_{p_\theta(x)} \left[ (\sum_{i=1}^N x_i  - N\theta)^2 \right] }{\theta^2 (1-\theta)^2}  = \frac{Var_{p_\theta(x)} \left( \sum_{i=1}^N x_i\right)  }{\theta^2 (1-\theta)^2} = \frac{N \theta (1-\theta)}{\theta^2 (1-\theta)^2} = \frac{N}{\theta (1-\theta)}$$

So that's cool, we can detect several properties of it. First of all the Fisher infomration increases linearly with the number of datapoints, we consider. Make's sense with more i.i.d. samples we learn more about the unknown parameter and each sample should contain the same amount of information. As we will later see, this will also be a universal property. Further we see that it is inversly proportianl to the variance of the distribution. This also makes sense that we can learn more about the unknown parameter from random variables with small variance, as we will see this property is to some degree also generalizable (but it is not that obvious as one would think intuitvely!). 

PLOT IT!



## Nice properties of the Fisher information

As already observed in the previous example. If we observe i.i.d. samples, each sample should contain the same amount of information. 

> **Claim:** Be $X$ and $Y$ be two independent random variables, then
> $$ F_{X,Y}(\theta) = F_{X}(\theta) + F_{Y}(\theta)$$
> **Proof:** We can write
> $$ \begin{aligned}
    F_{X,Y}(\theta) &= \mathbb{E}_{X,Y}\left[ s_{X,Y}(\theta)s_{X,Y}(\theta)^T \right]\\ 
    & = \int \int  \nabla_\theta \log p(x,y|\theta) \left( \nabla_\theta \log p(x,y|\theta) \right)^T p(x)p(y)dxdy\\
    &= \int \int  \nabla_\theta (\log p(x|\theta) + \log p(y|\theta)) \left( \nabla_\theta (\log p(x|\theta) + \log p(y|\theta)) \right)^T p(x)p(y)dxdy\\ 
    &= \int \nabla_\theta \log p(x|\theta)(\nabla_\theta \log p(x|\theta))^Tp(x) dx + \int \nabla_\theta \log p(y|\theta)(\nabla_\theta \log p(y|\theta))^Tp(y) dx\\ 
    & \quad +  \int \int \nabla_\theta \log p(x|\theta) (\nabla_\theta \log p(y|\theta))^T p(x)p(y)dxdy\\ 
    & \quad +  \int \int \nabla_\theta \log p(y|\theta) (\nabla_\theta \log p(x|\theta))^T p(y)p(x)dydx\\ 
    &= \mathbb{E}_{X}[s_X(\theta) s_X(\theta)^T] + \mathbb{E}_{Y}[s_Y(\theta) s_Y(\theta)^T] + \int  \underbrace{\mathbb{E}_X[ s_{X}(\theta)]}_{= 0} s_Y(\theta)^T p(y)dy + \int  \underbrace{\mathbb{E}_Y[ s_{Y}(\theta)]}_{= 0} s_X(\theta)^T p(x)dx\\ 
    & = F_X(\theta) + F_Y(\theta) 
\end{aligned}$$

So this is great and by it we can easily explain how the Fisher information behaves under the i.i.d. assumpiton:
$$ F_{X_1, \dots, X_n}(\theta) = \sum_{i=1}^N F_{X_i}(\theta) = N \cdot F_X(\theta)$$

We can extend this Lemma to a more general case, were $X$ and $Y$ are not independent:

> **Claim**: Be $X$ and $Y$ two random variables, jointly distributed according to $p(X,Y)$. Then 
> $$ F_{X,Y}(\theta) = F_{X|Y}(\theta) + F_Y(\theta) = F_{Y|X}(\theta) + F_X(\theta)$$
> Where we define the conditional Fisher information as
> $$ F_{X|Y} = \mathbb{E}_Y\left[ F_{X|y = Y}(\theta) \right] = \mathbb{E}_Y\left[ \int \nabla_\theta \log p(x|y = Y) (\nabla_\theta \log p(x|y = Y))^T p(x|y=Y)  \right]$$
> **Proof**: As above, but using the chain rule of probability $\log p(x,y) = \log p(x|y) + p(y)$. 

Within the Bernoulli example we already observed an close relationship to the variance of the random variable. This relationship is not as universal as you would guess! 



NOT SURE IF THERE ACTUALLY IS ONE ...

Another very usefull property is the connection to the hessian of the log likelihood. 

> **Claim:** Be $\log p_\theta(x)$ is twice differentiable and be $H_{\log p_\theta (x)}$ the corresponding Hessian. Under certain regularity conditions it holds that
> $$ F_X(\theta) = -\mathbb{E}_{p_\theta (x)} \left[ H_{\log p_\theta (x)}  \right]$$
> **Proof:** We can write the Hessian of the log likelihood as follows
> $$ H_{\log p_\theta (x)}= \nabla_\theta \nabla_\theta^T \log p_\theta (x) = \nabla_\theta \frac{\nabla_\theta^T p_\theta(x)}{p_\theta (x)} = \frac{ \nabla_\theta \nabla_\theta^T p_\theta (x) p_\theta(x) - \nabla_\theta p_\theta (x) \nabla_\theta^T p_\theta (x)}{p_\theta(x)^2} = \frac{H_{ p_\theta (x)}}{p_\theta (x)} - \frac{\nabla_\theta p_\theta (x) \nabla_\theta^T p_\theta (x)}{p_\theta (x)^2} $$
> Which simply follows from the chain and quotient rule of differentiation. Let's apply the expectation to the first term.
> $$ \mathbb{E}_{p_\theta (x)} \left[  \frac{H_{ p_\theta (x)}}{p_\theta (x)}  \right] = \int H_{ p_\theta (x)} dx = \int \nabla_\theta \nabla_\theta^T p_\theta (x)dx = \nabla_\theta \nabla_\theta^T \int p_\theta (x) dx = \nabla_\theta \nabla_\theta^T 1 = 0$$
> So this term fanishes, let's apply it to the second term
> $$ \mathbb{E}_{p_\theta (x)} \left[ - \frac{\nabla_\theta p_\theta (x) \nabla_\theta^T p_\theta (x)}{p_\theta (x)^2} \right] = - \mathbb{E}_{p_\theta (x)} \left[ \left( \frac{\nabla_\theta p_\theta (x)}{p_\theta (x)}\right) \left( \frac{\nabla_\theta p_\theta (x)}{p_\theta (x)}\right)^T \right] = - \mathbb{E}_{p_\theta (x)} \left[ \left( \nabla_\theta \log p_\theta (x)\right) \left( \nabla_\theta \log p_\theta (x)\right)^T \right]$$
> which proves the statement.

This interpretations also gives use a bunch of additional properties, 

> **Claim:** The Fisher information satisfies the following properties
> * $F(\theta)$ is symmetric and positive semi-definite.
> * $F(\theta)$ is positive definite if the statistical model is identifiable (there cannot exist two parameters $\theta_i \neq \theta_j$ such that $p_{\theta_i} = p_{\theta_j}$)
> * If $[\theta]_i$ and $[\theta]_j$ are independent then $[F(\theta)]_{ij} = 0$ i.e. then $[\theta]_i$ and $[\theta]_j$ can be estimated independently of each other.
> * Be $\eta = f(\theta)$ an alternative reparameterization, then $F(\eta) = J_f^T F(\theta) J_f$, where $J_f$ denotes the Jacobian matrix of $f$.
> 
> **Proof:**: You can do this, or I some day in the future ...



## Some applications in frequentist statistics...

Already within the introduction we closely related the Fisher Information to maximum likelihood estimation. Recall that in frequentist statistic we typically have to do the following:

* Propose an estimator (typically a point estiamte) of the parameter.
* Test wheather it's value aligns with the data.
* Derive confidence intervals.

As we will see in each of this steps the Fisher information will be involved in some way or another.

### Proposing an estimator

An estimator is typically just a function $f : X^N \rightarrow \Theta$, which maps a set of observations to a specific parameter. Yet in certain cases it may be less practical to work with the raw data and we instead want to preprocess it. We may even use a *statistic* to summarize the data, i.e. $T: X^N \rightarrow \mathcal{T}$. This may does simplify the design of an estiamtor significantly as we now only have to consider a single or a few statistics. Yet, do we actually use information by just considering the statistic? The most information should always be within the raw data, right? In fact this is indeed true (according to Fisher information) !

> **Claim:** (Data-processing inequality) Be $X$ an random variable and be $T(X)$ an arbitrary transformation independent of $\theta$. Then
> $$ F_{T(X)}(\theta) \leq F_X(\theta)$$
> Where for matrices we use the standard Loewner order i.e. $A \leq B \iff A - B$ is postive semi definite. The inequality is a equality if and only if $T$ is a *sufficient statistic*.
> 
> **Proof:** As the Fisher information satisfies the additive chain rule it follows that for any random variable $Y$ we have that
> $$ F_{Y, X}(\theta) = F_{Y| X}(\theta) + F_{X}(\theta) \geq F_X(\theta)$$
> where the inequality holds because $F_{Y| X}(\theta)$ is postive semi definite.
> Thus if follows that
> $$ F_{T(X)}(\theta) \leq F_{T(X), X}(\theta) = \underbrace{F_{T(X)|X}(\theta)}_{=0} + F_X(\theta) = F_X(\theta).$$
> Here $F_{T(X)|X}(\theta) = 0$ as $T(X)$ is a deterministic transform of $X$ and thus independent of $\theta$ given $X$.

So that's nice that Fisher informations does indeed follow our intuition. So let's start to actually do inference. The most common type to obtain an estimator $f$ of data is the maximum likelihood method. We typically propse some family of parameteric models $\mathcal{F} =\{ p_\theta | \theta \in \Theta \}$, given some data $x_1, ..., x_n \sim p^*$ we then select 
$$ \theta^* = \argmax_{\theta \in \Theta} \sum_{i=1}^n \log p_\theta(x_i)$$

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
