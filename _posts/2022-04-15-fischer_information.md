---
layout: article
title: A tale of Fischer information in ML
key: A4
tags: Statistics, Math
aside:
    toc: true
comment: true
cover: https://s8.gifyu.com/images/Download.png
---

Fischer information plays a pivotal role in machine learning, as we will see in some way or another it will pop up in both Frequentist or Bayesian statistical paradigms. In this post, I will introduce the Fisher information together with some important properties. We will go through several applications and relationships of the Fischer information to other important quantities.

## Notation and Definition

Suppose we have a parametric statistical model $p_{\theta}(x)$ with parameter vector $\theta$ modeling some distribution. Our goal is to learn an unknown distribution $p(x)$ from which we have i.i.d. samples $x_1, \dots, x_N \sim p(x)$. In frequentist statistics the by far most common approach to learn $\theta$ is by maximizing the likelihood $\prod_{i=1}^N p_\theta(x_i)$ with respect to the parameter $\theta$. To assess the goodness of fit we can use the so-called (Fischer) score, which we define as

$$ s_x(\theta) = \nabla_\theta \log p_\theta(x).$$

To justify this as a measure of goodness, consider the following claim:

[theorem]
Be $\log p_\theta(x)$ differentiable in $\theta$ almost everywhere and let the support of $p_\theta$ be independent of $\theta$. Then it holds that
$$ \mathbb{E}_{p_\theta(x)} \left[s_x(\theta)\right] =  \mathbb{E}_{p_\theta(x)}  \left[ \nabla_\theta \log p_\theta(x)\right] = 0 $$
[/theorem]
[proof]
As we can interchange integration and differentiation under the above regularity conditions it holds that
$$ \begin{aligned}\mathbb{E}_{p_\theta(x)}  \left[ \nabla_\theta \log p_\theta(x)\right]  &= \int \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} p_\theta(x)dx \\ &= \int \nabla_\theta q (x)dx \\&= \nabla_\theta \int p_\theta(x)dx \\&= \nabla_\theta 1\\& = 0 \end{aligned} $$
[/proof]

Thus assuming we found some nice estimator $\theta^*$, we would expect that if it is a good fit we have that

$$ \mathbb{E}_{p(x)}\left[ s_{x}(\theta)\right] \approx \frac{1}{N} \sum_{i=1}^N \nabla_{\theta^*} \log p_{\theta^*}(x_i) \approx 0$$

So that's great, as $N \rightarrow \infty$ this will be zero if and only if $p_{\theta^*} = p$! Yet, anyway, we are more interested in the deviation from zero to evaluate the goodness of fit. Thus let's look at the covariance of the score under the model. This idea leads us to the main topic of this post: The Fisher information (matrix)

<dir class="definition">
<b>(Fisher information):</b> Assuming the score $s(\theta)$ has a bounded second moment under $p_\theta$, then the the Fisher information (matrix) is defined as
$$F(\theta) := \Sigma_{q} \left(  s_x(\theta) \right) = \mathbb{E}_{p_\theta(x)} \left[ s_x(\theta)s_x(\theta)^T \right]$$
</dir>

We can again approximate the expectation of the empirical data distribution, yielding the empirical Fisher $\hat{F}(\theta)$, which we can again evaluate at our selected parameter $\theta$

$$ \hat{F}(\theta) = \frac{1}{N} \sum_{i=1}^N s_{x_i}(\theta) s_{x_i}(\theta)^T $$

Note, that this estimate still measures the deviation of the score from zero, not from the actual mean (the mean is only zero if $p = p_\theta$! Hence this estimate is only unbiased if $x_i \sim p_\theta$. In any case, if a random variable has high Fisher information it implies that the absolute value of the score is often large.

A quick look at [wikipedia](https://en.wikipedia.org/wiki/Fisher_information#Regularity_conditions) gives us the following intuitive definition
<center> "In mathematical statistics, the Fisher information is a way of measuring the amount of information that an observable random variable $X$ carries about an unknown parameter $\theta$ of a distribution that models X."  </center>

So let's go through some examples 
### Example 1: Bernoulli model

Let's consider a coin flip experiment. The coin $X$ will be distributed according to a Bernoulli distribution 

$$Ber(x;\theta) = \theta^X (1-\theta)^{1-X}$$

Given $N$ observations $x_1, \dots, x_N \sim Ber(x;\theta)$, we can denote the loglikelihood as

$$ \log p(X|\theta) = \sum_{i=1}^N \log p(x_i|\theta) = (\sum_{i=1}^N x_i)\log \theta + (N - \sum_{i=1}^N x_i) \log (1-\theta) $$

Thus let's first derive the score function of this problem
$$ s_X(\theta) = \nabla_\theta  \log p(X|\theta) = \frac{(\sum_{i=1}^N x_i)}{\theta} - \frac{(N - \sum_{i=1}^N x_i)}{1-\theta} = \frac{\sum_{i=1}^N x_i (1-\theta) - N\theta + \sum_{i=1}^N x_i \theta}{\theta(1-\theta)} = \frac{\sum_{i=1}^N x_i  - N\theta}{\theta (1-\theta)}$$

Notice that the score is zero if the sum of Bernoulli trials equals the expected i.e. $N\theta$, a consequence of Theorem 1. Furthermore the magnitude of the score is determined by $1/(\theta (1-\theta))$ which is minimized if $\theta=0.5$ and approaches infinity for $\theta \rightarrow 1$ and $\theta \rightarrow 0$. As a result changes within the parameter near zero or one will lead to a large score! Recall that the score equals the gradient of the log-likelihood. A large score thus indicates that small changes within the parameter can strongly change the log-likelihood.

So let's compute the corresponding Fisher information, as we defined it before

$$F(\theta) = \mathbb{E}_{p_\theta(x)} \left[ s(\theta)^2) \right] = \frac{\mathbb{E}_{p_\theta(x)} \left[ (\sum_{i=1}^N x_i  - N\theta)^2 \right] }{\theta^2 (1-\theta)^2}  = \frac{Var_{p_\theta(x)} \left( \sum_{i=1}^N x_i\right)  }{\theta^2 (1-\theta)^2} = \frac{N \theta (1-\theta)}{\theta^2 (1-\theta)^2} = \frac{N}{\theta (1-\theta)}$$

So that's cool, we can detect several properties of it. First of all the Fisher information increases linearly with the number of data points, we consider. Make does make sense with more i.i.d. samples we learn more about the unknown parameter and each sample should contain the same amount of information. As we will later see, this will also be a universal property. Further, we see that it is inversely proportional to the variance of the distribution. This also makes sense that we can learn more about the unknown parameter from random variables with small variance, as we will see this property is to some degree also generalizable (but it is not as obvious as one would think intuitively!). 


<p align="center">
  <img src="../../../assets/images/bernoulli_fisher_info.png" />
</p>



### Example 2: Exponential family

Let's look also at a more general family, which includes the previous example: The exponential family. Be $T(x)$ the sufficient statistics, $\eta$ the natural parameters, and $Z,h$ the partition function and the base measure. Then an exponential family model is defined as 

$$ p_\eta (x) = h(x) \exp(\eta^T T(x) - Z(\eta)) $$

with $Z(\eta) = \log \int h(x) \exp(\eta^TT(x)) dx$. The log-likelihood is thus for $N$ observations $x_1, ..., x_N$ is thus given by

$$ \log p_\eta(X) = \sum_{i=1}^N \eta^T T(x_i) - Z(\eta) + \log h(x_i)$$

For notational simplicity lets assume $N=1$, then the score of an exponential family is given by

$$ s_x(\eta) = \nabla_\eta \log p_\eta(x) = T(x) - \nabla_\eta Z(eta)$$

Notice that by this and Theorem 1 it follows that $\mathbb{E}_{p_\eta}[T(x)] = \nabla_\eta Z(eta)$ a nice property of exponential families. Let's get to the Fisher information 

$$ F(\eta) = \mathbb{E}_{p_\eta} \left[ (T(x) - \nabla_\eta Z(eta))(T(x) - \nabla_\eta Z(eta))^T   \right] = \Sigma_{p_\eta(x_i)}(T(x))$$

Thus the Fisher information equals the covariance of the sufficient statistic $T(x)$! But wait the Bernoulli distribution is an exponential family, right? And we just calculated that in this case, the Fisher information equals the inverse variance. So isn't this a contradiction?

No, we just discovered that Fisher information can change upon a change in parameterization. Remember, before the parameter $\theta$ was the probability of success. In contrast the natural parameter for a Bernoulli distribution is given by $\eta = \log \frac{\theta}{1 - \theta}$. But luckily as we will later see, translate between different parameterizations rather easily.

## Important properties of the Fisher information

If we observe i.i.d. samples, each sample should contain the same amount of information. 

<dir class="lemma">
Be $X$ and $Y$ be two independent random variables, then

$$ F_{X,Y}(\theta) = F_{X}(\theta) + F_{Y}(\theta)$$
</dir>
[proof]
We can write
$$ \begin{aligned}
    F_{X,Y}(\theta) &= \mathbb{E}_{X,Y}\left[ s_{X,Y}(\theta)s_{X,Y}(\theta)^T \right]\\ 
    & = \int \int  \nabla_\theta \log p(x,y|\theta) \left( \nabla_\theta \log p(x,y|\theta) \right)^T p(x)p(y)dxdy\\
    &= \int \int  \nabla_\theta (\log p(x|\theta) + \log p(y|\theta)) \left( \nabla_\theta (\log p(x|\theta) + \log p(y|\theta)) \right)^T p(x)p(y)dxdy\\ 
    &= \int \nabla_\theta \log p(x|\theta)(\nabla_\theta \log p(x|\theta))^Tp(x) dx + \int \nabla_\theta \log p(y|\theta)(\nabla_\theta \log p(y|\theta))^Tp(y) dx\\ 
    & \quad +  \int \int \nabla_\theta \log p(x|\theta) (\nabla_\theta \log p(y|\theta))^T p(x)p(y)dxdy\\ 
    & \quad +  \int \int \nabla_\theta \log p(y|\theta) (\nabla_\theta \log p(x|\theta))^T p(y)p(x)dydx\\ 
    &= \mathbb{E}_{X}[s_X(\theta) s_X(\theta)^T] + \mathbb{E}_{Y}[s_Y(\theta) s_Y(\theta)^T] + \int  \underbrace{\mathbb{E}_X[ s_{X}(\theta)]}_{= 0} s_Y(\theta)^T p(y)dy + \int  \underbrace{\mathbb{E}_Y[ s_{Y}(\theta)]}_{= 0} s_X(\theta)^T p(x)dx\\ 
    & = F_X(\theta) + F_Y(\theta) 
\end{aligned}$$
[/proof]

So this is great and by it, we can easily explain how the Fisher information behaves under the i.i.d. assumption:

<dir class="theorem">
If $X_1, \dots, X_N$ are i.i.d. then

$$ F_{X_1, \dots, X_n}(\theta) = N \cdot F_{X_1}(\theta)$$
</dir>

We can extend Lemma 1 to a more general case, where $X$ and $Y$ are not independent:

<dir class="lemma">
Be $X$ and $Y$ two random variables, jointly distributed according to $p(X,Y)$. Then 

$$ F_{X,Y}(\theta) = F_{X|Y}(\theta) + F_Y(\theta) = F_{Y|X}(\theta) + F_X(\theta)$$

where we define the conditional Fisher information as

$$ F_{X|Y} = \mathbb{E}_Y\left[ F_{X|y = Y}(\theta) \right] = \mathbb{E}_Y\left[ \int \nabla_\theta \log p(x|y = Y) (\nabla_\theta \log p(x|y = Y))^T p(x|y=Y)  \right]$$
</dir>
[proof]
As above just using the product rule $\log p(X,Y) = \log p(X|Y) + \log p(Y) = \log p(Y|X) + \log p(X)$
[/proof]

We already saw that the Fisher information depends on the parameterization of our statistical model. So let's try to exactly quantify how different parameterizations do change the Fisher information.

<dir class="lemma">
Be f a totally differentiable map from parameter with $\eta = f(\theta)$, where both parameters encode the same statistical model. Be $J_f = \nabla_\theta f(\theta)$ the Jacobian matrix, then
$$ F_{x}(\theta) = J_f^T F_X(f(\theta)) J_f $$
</dir>
[proof]
This simply follows from the chain rule of differentiation. Notice that we can write
\[$$\displaystyle \begin{aligned}s_x(\theta)= \nabla_\theta \log p_{f(\theta)}(x) = \nabla_\theta f(\theta) \nabla_\eta \log p_\eta(x) = J_f^T \nabla_\eta \log p_\eta(x) = J_f^T s_x(\eta) \end{aligned}$$ <br />
By the definition of Fisher information, we have that 
$$ F_x(\theta) = \mathbb{E}_{p_\theta}\left[s_x(\theta)s_x(\theta)^T \right] = \mathbb{E}_{p_\theta}\left[J_f^Ts_x(\eta)s_x(\eta)^TJ_f \right] = J_f^T F_x(\eta) J_f$$ <br />
which completes the proof.
[/proof]

Recall that the natural parameters of a Bernoulli distribution are $\eta = f(\theta) = \log \frac{\theta}{1-\theta}$. Thus $\nabla_\theta f(\theta) = \frac{1}{\theta(1-\theta)}$, we had that the Fisher information of an exponential family equals the covariance of the test statistic. For a Bernoulli variable, we thus have

$$  F_x(\eta) = f^{-1}(\eta) (1- f^{-1}/\eta) = \theta (1-\theta)$$

Thus by applying the above lemma we get
$$ F_x(\theta) = \frac{\theta (1-\theta)}{(\theta (1-\theta)^2} = \frac{1}{\theta (1-\theta)}$$
which is exactly what we obtained by explicitly calculating it.

Another very useful property is the connection to the Hessian of the log-likelihood. This is often used as an alternative definition and can also simplify the computation. 

<dir class="theorem">
 Be $\log p_\theta(x)$ twice differentiable and be $H_{\log q (x)} = \nabla_\theta \nabla_\theta^T \log p_\theta(x)$ the corresponding Hessian. Then it holds that
 
 $$ F_X(\theta) = -\mathbb{E}_{p_\theta (x)} \left[ H_{\log q (x)}  \right]$$
 </dir>
 [proof]
  We can write the Hessian of the log-likelihood as follows
 $$ H_{\log q (x)}= \nabla_\theta \nabla_\theta^T \log q (x) = \nabla_\theta \frac{\nabla_\theta^T p_\theta(x)}{q (x)} = \frac{ \nabla_\theta \nabla_\theta^T q (x) p_\theta(x) - \nabla_\theta q (x) \nabla_\theta^T q (x)}{p_\theta(x)^2} = \frac{H_{ q (x)}}{q (x)} - \frac{\nabla_\theta q (x) \nabla_\theta^T q (x)}{q (x)^2} $$ <br />
 Which simply follows from the chain and quotient rule of differentiation. Let's apply the expectation to the first term.
 $$ \mathbb{E}_{q (x)} \left[  \frac{H_{ q (x)}}{q (x)}  \right] = \int H_{ q (x)} dx = \int \nabla_\theta \nabla_\theta^T q (x)dx = \nabla_\theta \nabla_\theta^T \int q (x) dx = \nabla_\theta \nabla_\theta^T 1 = 0$$ <br />
 So this term vanishes, let's apply it to the second term
 $$ \mathbb{E}_{q (x)} \left[ - \frac{\nabla_\theta q (x) \nabla_\theta^T q (x)}{q (x)^2} \right] = - \mathbb{E}_{q (x)} \left[ \left( \frac{\nabla_\theta q (x)}{q (x)}\right) \left( \frac{\nabla_\theta q (x)}{q (x)}\right)^T \right] = - \mathbb{E}_{q (x)} \left[ \left( \nabla_\theta \log q (x)\right) \left( \nabla_\theta \log q (x)\right)^T \right]$$ <br />
 which proves the statement.
[/proof]


This allows us to interpret the Fisher information as the average curvature of the log-likelihood. Low Fisher information thus indicates maxima's are shallow and many nearby values have similar log-likelihood. Conversely, high Fisher information indicates a sharp maxima i.e. changes within the parameters have a strong influence on the log-likelihood. So let's investigate this visually. Recall the Fisher information should represent how "easy" we can learn an unknown parameter from data.

<p align="center">
<figure>
  <img src="../../../assets/normal_high_variance.png" />
  <img src="../../../assets/normal_low_variance.png" />
  <img src="../../../assets/student_t.png" />
  <figcaption> On the left I plot the log-likelihood functions evaluated at several samples $x_i \sim \mathcal{N}(x;\mu^*, 25)$. The red bar indicates the ground truth parameter $\mu^* = 5$. The central plots show the distribution of scores, with estimated empirical variance. According to Theorem 1, the mean must be zero. The last plot shows the distribution of second derivatives, note that for Gaussian cases this is constant. The orange bar indicates the expected value. According to Theorem 3, this corresponds to the negative variance of the score.  </figcaption>
</figure>
</p>



This leads us to the sort of distance measure between probability measures and in fact, the Fisher information has a close connection to f-divergence, especially the KL divergence.

<dir class="theorem">
Be $p_\theta(x)$ some statistical model with Fisher information matrix $F_x(\theta)$ on parameter space $\theta$. Then for any $ \delta \in \theta $ it follows that
$$ D_{KL}(p_\theta(x)||p_{\theta + \delta}(x)) = \frac{1}{2} \delta^T F_x(\theta) \delta + o(||\delta||^2)$$
</dir>
[proof]
It follows to form a second order Taylor expansion around $\theta$. For notational simplicity be $\theta' = \theta + \delta$, then <br />
$$ \begin{aligned} D_{KL}(p_\theta(x) ||  p_{\theta'}(x)) \approx \ & D_{KL}(p_\theta(x) || p_{\theta}(x)) \ + \\ &(\nabla_{\theta'}^TD_{KL}(p_\theta || p_{\theta'})\mid_{\theta'=\theta}) (\theta' - \theta) \ + \\ &\frac{1}{2}(\theta' - \theta)^T\left(\nabla_{\theta'}\nabla_{\theta'}^TD_{KL}(p_\theta || p_{\theta'})\mid_{\theta'=\theta}\right) (\theta' - \theta) \end{aligned} $$ <br />
By definition $\theta' - \theta) = \delta$. Further, the first term vanishes once evaluated at $\theta' = \theta$, by the properties of a divergence.  For the first order term, we have <br />
$$ \begin{aligned} \nabla_{\theta'}D_{KL}(p_\theta || p_{\theta'}) & = \nabla_{\theta'} \mathbb{E}_{p_\theta} \left[ \log \frac{p_{\theta}(x)}{p_{\theta'}(x)} \right] \\ &= \mathbb{E}_{p_\theta} \left[ \nabla_{\theta'}\log p_{\theta}(x) - \nabla_{\theta'}\log p_{\theta'}(x) \right] \\&= -\mathbb{E}_{p_\theta} \left[ \nabla_{\theta'}\log p_{\theta'}(x) \right] \end{aligned}$$ <br />
which vanishes once we evaluate the term at $\theta' = \theta$, due to Theorem 1. So only the second order term remains. Be $H_{\log p_{\theta'}(x)} = \nabla_{\theta'}^T\nabla_{\theta'} \log p_{\theta'}(x)$ the Hessian matrix. Then we can write Taylor expansion as follows <br />
$$ D_{KL}(p_\theta(x) || p_{\theta + \delta}(x)) = \frac{1}{2} \delta^T \mathbb{E}_{p_\theta(x)}\left[  -H_{\log p_{\theta'}(x)} \mid_{\theta' = \theta}\right] \delta + o(||\delta||^2) = \delta^T F_x(\theta) \delta + o(||\delta||^2)$$ <br />
which proves the statement.
[/proof]

Within the Bernoulli example, we already observed a close relationship to the variance of the random variable. This relationship is not as universal as you would guess! At least within the exponential family, there is a clear connection to the covariance of the test statistic. Yet, recall that this is only for the natural parameters. In general, we can relate these quantities through a lower bound: 

And here are some other nice properties
<dir class=theorem>
 The Fisher information satisfies the following properties <br />
  (i)  $F(\theta)$ is symmetric and positive semi-definite. <br />
  (ii) $F(\theta)$ is positive definite if the statistical model is identifiable (there cannot exist two parameters $\theta_i \neq \theta_j$ such that $p_{\theta_i} = p_{\theta_j}$)<br />
  (iii) If $[\theta]_i$ and $[\theta]_j$ are independent then $[F(\theta)]_{ij} = 0$ i.e. then $[\theta]_i$ and $[\theta]_j$ can be estimated separately <br />
</dir>
[proof]
 Proof:: You can do this, or I someday in the future ...
[/proof]

## Some applications in frequentist statistics

Already within the introduction we closely related the Fisher Information to maximum likelihood estimation. Recall that in frequentist statistics we typically have to do the following:

 * Propose an estimator (typically a point estimate) of the parameter.
 * Test whether its value aligns with the data.
 * Derive confidence intervals.

As we will see in each of these steps the Fisher information will be involved in some way or another.

### Proposing an estimator

An estimator is typically just a function $f : X^N \rightarrow \theta$, which maps a set of observations to a specific parameter. Yet in certain cases, it may be less practical to work with the raw data and we instead want to preprocess it. We may even use a statistic to summarize the data, i.e. $T: X^N \rightarrow \mathcal{T}$. This may simplify the design of an estimator significantly as we now only have to consider a single or a few statistics. Yet, do we lose information by just considering the statistic? Most information should always be within the raw data, right? This is indeed true (according to Fisher information)!

<div class="theorem">
 (Data-processing inequality):   Be $X$ a random variable and be $T(X)$ an arbitrary transformation independent of $\theta$. Then
 $$ F_{T(X)}(\theta) \leq F_X(\theta)$$
Where for matrices we use the standard Loewner order i.e. $A \leq B \iff A - B$ is positive semi-definite. The inequality is equality if and only if $T$ is a sufficient statistic.
</div>
[proof]
As the Fisher information satisfies the additive chain rule it follows that for any random variable $Y$ we have that <br />
$$ F_{Y, X}(\theta) = F_{Y| X}(\theta) + F_{X}(\theta) \geq F_X(\theta)$$ <br />
where the inequality holds because $F_{Y| X}(\theta)$ is positive semi-definite. Thus it follows that <br />
$$ F_{T(X)}(\theta) \leq F_{T(X), X}(\theta) = \underbrace{F_{T(X)|X}(\theta)}_{=0} + F_X(\theta) = F_X(\theta).$$ <br />
Here $F_{T(X)|X}(\theta) = 0$ as $T(X)$ is a deterministic transform of $X$ and thus independent of $\theta$ given $X$.
[/proof]

So that's nice that Fisher's information does indeed follow our intuition. So let's start to do inference. The most common type to obtain an estimator $f$ of data is the maximum likelihood method. We typically propose some family of parametric models $\mathcal{F} =\{ p_\theta \mid \theta \in \theta \}$, given some data $x_1, \cdots, x_n \sim p$ we then select 

$$ \hat{\theta} = \arg\max_{\theta \in \theta} \sum_{i=1}^n \log p_\theta(x_i)$$

Note as $X_1, \cdots, X_n$ are independent realization of random variables following measure $p$. The estimate $\hat{\theta}$ is thus itself a random variable. From major interest are the moments of this random variable, which are used to define two important measures of quality:
* **Bias**: A well-behaved estimator should at least be in expectation correct. This is quantified by the bias
  $$ b(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta^* $$
  where $\theta^*$ denotes the "true" (or best achievable) parameter. A parameter with zero bias is called *unbiased*.
* **Variance**: A unbiased point estimate is nice, but inefficient if it has high variance i.e. if it is widely spread around the mean then by performing a single point estimate you will likely land far away from it and so from the true parameter. 

As introduced there is a close connection not only for the MLE estimator but for any point estimate. While it is generally very hard to compute the exact variance of an estimator. We can derive a lower bound the *Carmér Rao lower bound*, given by

<div class="theorem">
Be $\theta^* \in \theta$ the parameters of a probability density $p_{\theta^*}$. Assume that the Fisher information matrix $F_x(\theta)$ exists. Be $g(X)$ an statistical estimator of $\theta^*$ and assume that the first moment $\psi(\theta^*) =\mathbb{E}_{p_{\theta^*}}[g(X)]$ exists. Then the covariance of the estimator satisfies 

$$ \Sigma_{p_{\theta^*}}(g(X)) \geq \nabla_{\theta^*}\psi(\theta^*) \left[ F(\theta^*) \right]^{-1} \nabla_{\theta^*}\psi(\theta^*)^T$$

If the bias $b(\theta^*)$ is known. Then we can also rewrite it as

$$ \Sigma_{p_{\theta^*}}(g(X)) \geq (1 + b(\theta^*)) \left[ F(\theta^*) \right]^{-1} (1 + b'(\theta^*))^T$$

If the estimator is unbiased, then 

$$ \Sigma_{p_{\theta^*}}(g(X)) \geq F(\theta^*)^{-1}$$

</div>
[proof]
test
[/proof]

This is nice because it lets us evaluate the efficiency of estimators. Hence we call an estimator *efficient* if the variance equals the Cramér Rao lower bound.

In general, the distribution of an arbitrary statistical estimator can become very complicated. Yet, to be able to run statistical tests or construct confidence intervals, we have to know the actual distribution. Even for the "nice" MLE, this can be generally hard but at least good asymptotic results exist

<dir class="theorem">
(Asymptotic normality of the MLE): Be $X_1, \dots, X_n \sim p_{\theta^*}$ and be $\hat{\theta}$ the MLE of $\theta^*$. Then 

$$ \sqrt{n}(\hat{\theta} - \theta^*) \rightarrow \mathcal{N}(\theta; 0, F(\theta^*)^{-1}) $$
</dir>
[proof]
test
[/proof]


### Experimental design

 By Theorem 8, we can estimate the standard deviation of the MLE by $\sigma_F = \sqrt{(n F_x(\theta^*))^{-1}}$ whenever $n$ is large enough. Observe that the standard deviation decreases whenever the Fisher information or $n$ is large. In practice, we cannot control the true value, but we can affect the number of trials we perform i.e. by collecting more data. So it may be interesting to know how many samples we need such that the MLE has a standard deviation below a certain threshold...

 Let's again consider a Bernoulli experiment. We want to estimate the probability $\theta^*$ using the MLE $\hat{\theta}$ but want to ensure that the standard deviation $\sigma_F \leq \epsilon$. It follows that we have to ensure that

 $$ \sqrt{(n F_x(\theta^*))^{-1}} \leq \epsilon \iff  n \geq \frac{1}{\epsilon^2}F_x(\theta^*)^{-1}$$

 Note that in practice we do not know the true value $\theta^*$, thus we may solve this problem for the worst-case scenario i.e

$$ n \geq \frac{1}{\epsilon^2} \max_\theta F_x(\theta)^{-1} =  \frac{1}{\epsilon^2} F_x(0.5)^{-1} = \frac{1}{(2\epsilon)^2}$$

As a result to achive a standard deviation of $\sigma_F \leq 0.1$ we need $n \geq 25$ samples.


### Testing hypothesis

This also paves the way to approximate the sampling distribution of any MLE of an arbitrary statistical model. Note the sampling distribution of an MLE is the distribution of estimates given independent realizations $X_1, \dots, X_N \sim p_{\theta^*}$. For example for large enough $n$ MLE estimates will fall into the range
$$ \left( \theta^* - 2.96 \sigma_F, \theta^* + 2.96 \right) $$
with (approximately) a $99%$ chance.

This is visually demonstrated in the figure below. Notice that even for small $n$ the approximation here is pretty good. Further as postulated by small Fisher information, it is harder to accurately estimate $\theta^* = 0.5$ than $\theta^*= 0.95$.

<p align="center">
<figure>
  <img src="../../../assets/bernoulli_confidence.png" />
  <img src="../../../assets/bernoulli_confidence2.png" />
  <figcaption> The figures show the sampling distribution of the MLE for a different number of data points, as well as the constructed $99\%$ confidence intervals. </figcaption>
</figure>
</p>

This allows us to construct a null hypothesis test for testing $\theta^* = \theta_0$. Let's say we want to test

$$ H_0: \theta^* = 0.5 \qquad H_1: \theta^* \neq 0.5$$

and choose an significance level of $\alpha = 0.01$. We know that for $n = 25$ trials, as a result $\sigma_F = 0.1$. Hence $99 \%$ of the time and MLE estimate would fall in between $(0.21, 0.79)$. Hence we would reject the null hypothesis if $\hat{\theta} < 0.21$ or $\hat{\theta} > 0.79$.
### Constructing confidence intervals

An alternative way to quantify our confidence in the estimate is to use confidence intervals. Recall that from the frequentist perspective the definition is a bit fuzzy: A $99\%$ confidence interval does not mean that the true value will lay in it with probability $0.99$, but that if we estimate 100 such intervals that 99 of them will contain the true value. We can again just replace the true value with our estimate $\hat{\theta}$. This is visually demonstrated in the plot below.

## Fisher information in Bayesian statistics

Until now we mostly discussed point estimates, so why is it also relevant in the Bayesian case? In the Bayesian paradigm, we start with existing prior beliefs about the parameter and update these beliefs based on information provided by data to guide inferential decisions.

Our prior beliefs are typically expressed in terms of a *prior* distribution $p(\theta)$, data is assumed to be generated by $p(x \mid \theta)$. Based on probability theory there is only a single way how observing $x$ should update our beliefs about our parameters, namely Baye's theorem

$$ p(\theta \mid x) = \frac{p(x \mid \theta)p(\theta)}{p(x)} $$

An intermediate connection to frequentist approach if now just start to do drop to full posterior in the fate of using only some point estimates derived of it. For example, we can use the posterior mean, variance, or mode. But let's try to stay Bayesian! As such we may want to extend the definition of the Fisher information matrix. 

<dir class="definition">
Given a generative model $p(x,\theta) = p(x|\theta)p(\theta)$. We define the Bayesian Fisher information matrix as 

$$ F = \mathbb{E}_{p(\theta)} \left[ F_x(\theta) \right] + \mathbb{E}_{p(\theta)} \left[\nabla_\theta \log p(\theta) \nabla_\theta \log p(\theta)^T \right]$$

where $ F_x(\theta)$ is the usual Fisher information as defined above. 
</dir>

Notice that in the Bayesian case, the Fisher information is constant for the parameter but does change with respect to the prior. It is easy to see that as $p(\theta) \rightarrow \delta (\theta_0)$ (i.e. the prior becomes a point mass) also $F \rightarrow F_x(\theta_0)$ i.e. we recover the frequentist Fisher information evaluated at $\theta_0$.

The intuitive meaning also changes a bit. Given some $x \sim p(x)$ the BFIM can be seen as the amount of information $x$ carries averaged over any possible parameter, as specified by the prior plus the information provided by the prior itself.

## Fisher information and the posterior



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
