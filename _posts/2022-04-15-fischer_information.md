---
layout: article
title: A tale of Fischer information in ML
key: A4
tags: Statistics, Math
aside:
    toc: true
comment: true
---

Fischer information plays a pivotal role in machine learning, as we will see in some way or another it will pop up in both Frequentist or Bayesian statistical paradigms. In this post, I will introduce the Fisher information together with some important properties. We will go through several applications and relationships of the Fischer information to other important quantities.

## Notation and Definition

Suppose we have a parametric statistical model $q_{\phi}(x)$ with parameter vector $\phi$ modeling some distribution. Our goal is to learn an unknown distribution $p(x)$ from which we have i.i.d. samples $x_1, \dots, x_N \sim p(x)$. In frequentist statistics the by far most common approach to learn $\phi$ is by maximizing the likelihood $\prod_{i=1}^N q_\phi(x_i)$ with respect to the parameter $\phi$. To assess the goodness of fit we can use the so-called (Fischer) score, which we define as

$$ s_x(\phi) = \nabla_\phi \log q_\phi(x).$$

To justify this as a measure of goodness, consider the following claim:

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

So that's great, as $N \rightarrow \infty$ this will be zero if and only if $q_{\phi^*} = p$! Yet, anyway, we are more interested in the deviation from zero to evaluate the goodness of fit. Thus let's look at the covariance of the score under the model. This idea leads us to the main topic of this post: The Fisher information (matrix)

<dir class="definition">
<b>(Fisher information):</b> Assuming the score $s(\phi)$ has a bounded second moment under $q_\phi$, then the the Fisher information (matrix) is defined as
$$F(\phi) := \Sigma_{q} \left(  s_x(\phi) \right) = \mathbb{E}_{q_\phi(x)} \left[ s_x(\phi)s_x(\phi)^T \right]$$
</dir>

We can again approximate the expectation of the empirical data distribution, yielding the empirical Fisher $\hat{F}(\phi)$, which we can again evaluate at our selected parameter $\phi$

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

Notice that the score is zero if the sum of Bernoulli trials equals the expected i.e. $N\phi$, a consequence of Theorem 1. Furthermore the magnitude of the score is determined by $1/(\phi (1-\phi))$ which is minimized if $\phi=0.5$ and approaches infinity for $\phi \rightarrow 1$ and $\phi \rightarrow 0$. As a result changes within the parameter near zero or one will lead to a large score! Recall that the score equals the gradient of the log-likelihood. A large score thus indicates that small changes within the parameter can strongly change the log-likelihood.

So let's compute the corresponding Fisher information, as we defined it before

$$F(\phi) = \mathbb{E}_{q_\phi(x)} \left[ s(\phi)^2) \right] = \frac{\mathbb{E}_{q_\phi(x)} \left[ (\sum_{i=1}^N x_i  - N\phi)^2 \right] }{\phi^2 (1-\phi)^2}  = \frac{Var_{q_\phi(x)} \left( \sum_{i=1}^N x_i\right)  }{\phi^2 (1-\phi)^2} = \frac{N \phi (1-\phi)}{\phi^2 (1-\phi)^2} = \frac{N}{\phi (1-\phi)}$$

So that's cool, we can detect several properties of it. First of all the Fisher information increases linearly with the number of data points, we consider. Make does make sense with more i.i.d. samples we learn more about the unknown parameter and each sample should contain the same amount of information. As we will later see, this will also be a universal property. Further, we see that it is inversely proportional to the variance of the distribution. This also makes sense that we can learn more about the unknown parameter from random variables with small variance, as we will see this property is to some degree also generalizable (but it is not as obvious as one would think intuitively!). 


<p align="center">
  <img src="../../../assets/images/bernoulli_fisher_info.png" />
</p>



### Example 2: Exponential family

Let's look also at a more general family, which includes the previous example: The exponential family. Be $T(x)$ the sufficient statistics, $\eta$ the natural parameters, and $Z,h$ the partition function and the base measure. Then an exponential family model is defined as 

$$ p_\eta (x) = h(x) \exp(\eta^T T(x) - Z(\eta)) $$

with $Z(\eta) = \log \int h(x) \exp(\eta^TT(x)) dx$. The log likelihood is thus for $N$ observations $x_1, ..., x_N$ is thus given by

$$ \log p_\eta(X) = \sum_{i=1}^N \eta^T T(x_i) - Z(\eta) + \log h(x_i)$$

For notational simplicity lets assume $N=1$, then the score of an exponential family is given by

$$ s_x(\eta) = \nabla_\eta \log p_\eta(x) = T(x) - \nabla_\eta Z(eta)$$

Notice that by this and Theorem 1 it follows that $\mathbb{E}_{p_\eta}[T(x)] = \nabla_\eta Z(eta)$ a nice property of exponential families. Let's get to the Fisher information 

$$ F(\eta) = \mathbb{E}_{p_\eta} \left[ (T(x) - \nabla_\eta Z(eta))(T(x) - \nabla_\eta Z(eta))^T   \right] = \Sigma_{p_\eta(x_i)}(T(x))$$

Thus the Fisher information equals the covariance of the sufficient statistic $T(x)$! But wait the Bernoulli distribution is an exponential family, right? And we just calculated that in this case, the Fisher information equals the inverse variance. So isn't this a contradiction?

No, we just discovered that Fisher information can change upon a change in parameterization. Remember, before the parameter $\phi$ was the probability of success. In contrast the natural parameter for a Bernoulli distribution is given by $\eta = \log \frac{\phi}{1 - \phi}$. But luckily as we will later see, translate between different parameterizations rather easily.

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

So this is great and by it, we can easily explain how the Fisher information behaves under the i.i.d. assumption:

<dir class="theorem">
If $X_1, \dots, X_N$ are i.i.d. then

$$ F_{X_1, \dots, X_n}(\phi) = N \cdot F_{X_1}(\phi)$$
</dir>

We can extend Lemma 1 to a more general case, where $X$ and $Y$ are not independent:

<dir class="lemma">
Be $X$ and $Y$ two random variables, jointly distributed according to $p(X,Y)$. Then 

$$ F_{X,Y}(\phi) = F_{X|Y}(\phi) + F_Y(\phi) = F_{Y|X}(\phi) + F_X(\phi)$$

where we define the conditional Fisher information as

$$ F_{X|Y} = \mathbb{E}_Y\left[ F_{X|y = Y}(\phi) \right] = \mathbb{E}_Y\left[ \int \nabla_\phi \log p(x|y = Y) (\nabla_\phi \log p(x|y = Y))^T p(x|y=Y)  \right]$$
</dir>
[proof]
As above just using the product rule $\log p(X,Y) = \log p(X|Y) + \log p(Y) = \log p(Y|X) + \log p(X)$
[/proof]

We already saw that the Fisher information depends on the parameterization of our statistical model. So let's try to exactly quantify how different parameterizations do change the Fisher information.

<dir class="lemma">
Be f a totally differentaibe map from parameter with $\eta = f(\phi)$, where both parameters encode the same statistical model. Be $J_f = \nabla_\phi f(\phi)$ the Jacobian matrix, then
$$ F_{x}(\phi) = J_f^T F_X(f(\phi)) J_f $$
</dir>
[proof]
This simply folows from the chain rule of differentiation. Notice that we can write
\[$$\displaystyle \begin{aligned}s_x(\phi)= \nabla_\phi \log q_{f(\phi)}(x) = \nabla_\phi f(\phi) \nabla_\eta \log q_\eta(x) = J_f^T \nabla_\eta \log q_\eta(x) = J_f^T s_x(\eta) \end{aligned}$$ <br />
By the definition of Fisher information we have that 
$$ F_x(\phi) = \mathbb{E}_{q_\phi}\left[s_x(\phi)s_x(\phi)^T \right] = \mathbb{E}_{q_\phi}\left[J_f^Ts_x(\eta)s_x(\eta)^TJ_f \right] = J_f^T F_x(\eta) J_f$$ <br />
which completes the proof.
[/proof]

Recall that the natural parameters of a Bernoulli distribution are $\eta = f(\phi) = \log \frac{\phi}{1-\phi}$. Thus $\nabla_\phi f(\phi) = \frac{1}{\phi(1-\phi)}$, we had that the Fisher information of an exponential family equals the covariance of the test statistic. For a Bernoulli variable we thus have

$$  F_x(\eta) = f^{-1}(\eta) (1- f^{-1}/\eta) = \phi (1-\phi)$$

Thus by applying the above lemma we get
$$ F_x(\phi) = \frac{\phi (1-\phi)}{(\phi (1-\phi)^2} = \frac{1}{\phi (1-\phi)}$$
which is exactly what we obtained by explicitly calculating it.

Another very useful property is the connection to the Hessian of the log-likelihood. This is often used as an alternative definition and can also simplify the computation. 

<dir class="theorem">
 Be $\log q_\phi(x)$ twice differentiable and be $H_{\log q (x)} = \nabla_\phi \nabla_\phi^T \log q_\phi(x)$ the corresponding Hessian. Then it holds that
 
 $$ F_X(\phi) = -\mathbb{E}_{q_\phi (x)} \left[ H_{\log q (x)}  \right]$$
 </dir>
 [proof]
  We can write the Hessian of the log likelihood as follows
 $$ H_{\log q (x)}= \nabla_\phi \nabla_\phi^T \log q (x) = \nabla_\phi \frac{\nabla_\phi^T q_\phi(x)}{q (x)} = \frac{ \nabla_\phi \nabla_\phi^T q (x) q_\phi(x) - \nabla_\phi q (x) \nabla_\phi^T q (x)}{q_\phi(x)^2} = \frac{H_{ q (x)}}{q (x)} - \frac{\nabla_\phi q (x) \nabla_\phi^T q (x)}{q (x)^2} $$ <br />
 Which simply follows from the chain and quotient rule of differentiation. Let's apply the expectation to the first term.
 $$ \mathbb{E}_{q (x)} \left[  \frac{H_{ q (x)}}{q (x)}  \right] = \int H_{ q (x)} dx = \int \nabla_\phi \nabla_\phi^T q (x)dx = \nabla_\phi \nabla_\phi^T \int q (x) dx = \nabla_\phi \nabla_\phi^T 1 = 0$$ <br />
 So this term vanishes, let's apply it to the second term
 $$ \mathbb{E}_{q (x)} \left[ - \frac{\nabla_\phi q (x) \nabla_\phi^T q (x)}{q (x)^2} \right] = - \mathbb{E}_{q (x)} \left[ \left( \frac{\nabla_\phi q (x)}{q (x)}\right) \left( \frac{\nabla_\phi q (x)}{q (x)}\right)^T \right] = - \mathbb{E}_{q (x)} \left[ \left( \nabla_\phi \log q (x)\right) \left( \nabla_\phi \log q (x)\right)^T \right]$$ <br />
 which proves the statement.
[/proof]


This allows us to interpret the Fisher information as the average curvature of the log-likelihood. Low Fisher information thus indicates maxima's are shallow and many nearby values have similar log-likelihood. Conversely, high Fisher information indicates a sharp maxima i.e. changes within the parameters have a strong influence on the log-likelihood. So let's investigate this visually. Recall the Fisher information should represent how "easy" we can learn an unknown parameter from data.

<p align="center">
<figure>
  <img src="../../../assets/normal_high_variance.png" />
  <img src="../../../assets/normal_low_variance.png" />
  <img src="../../../assets/student_t.png" />
  <figcaption> On the left I plot the log likelihood functions evaluated at several samples $x_i \sim \mathcal{N}(x;\mu^*, 25)$. The red bar indicates the ground truth parameter $\mu^* = 5$. The central plots show the distribution of scores, with estimated empirical variance. According to Theorem 1 the mean must be zero. The last plot shows the distribution of second derivatives, note that for Gaussian cases this is constant. The orange bar, indicates the expected value. According to Theorem 3 this corresponds to the negative variance of the score.  </figcaption>
</figure>
</p>



This leads us to the sort of distance measure between probability measures and in fact, the Fisher information has a close connection to f-divergence, especially the KL divergence.

<dir class="theorem">
Be $q_\phi(x)$ some statistical model with Fisher information matrix $F_x(\phi)$ on parameter space $\Phi$. Then for any $ \delta \in \Phi $ it follows that
$$ D_{KL}(q_\phi(x)||q_{\phi + \delta}(x)) = \frac{1}{2} \delta^T F_x(\phi) \delta + o(||\delta||^2)$$
</dir>
[proof]
It follows form a second order Taylor expansion around $\phi$. For notational simplicity be $\phi' = \phi + \delta$, then <br />
$$ \begin{aligned} D_{KL}(q_\phi(x) ||  q_{\phi'}(x)) \approx \ & D_{KL}(q_\phi(x) || q_{\phi}(x)) \ + \\ &(\nabla_{\phi'}^TD_{KL}(q_\phi || q_{\phi'})\mid_{\phi'=\phi}) (\phi' - \phi) \ + \\ &\frac{1}{2}(\phi' - \phi)^T\left(\nabla_{\phi'}\nabla_{\phi'}^TD_{KL}(q_\phi || q_{\phi'})\mid_{\phi'=\phi}\right) (\phi' - \phi) \end{aligned} $$ <br />
By definition $\phi' - \phi) = \delta$. Further the first term vanishes once evaluated at $\phi' = \phi$, by the properties of a divergence.  For the first order term we have <br />
$$ \begin{aligned} \nabla_{\phi'}D_{KL}(q_\phi || q_{\phi'}) & = \nabla_{\phi'} \mathbb{E}_{q_\phi} \left[ \log \frac{q_{\phi}(x)}{q_{\phi'}(x)} \right] \\ &= \mathbb{E}_{q_\phi} \left[ \nabla_{\phi'}\log q_{\phi}(x) - \nabla_{\phi'}\log q_{\phi'}(x) \right] \\&= -\mathbb{E}_{q_\phi} \left[ \nabla_{\phi'}\log q_{\phi'}(x) \right] \end{aligned}$$ <br />
which vanishes once we evaluate the term at $\phi' = \phi$, due to Theorem 1. So only the second order term remains. Be $H_{\log q_{\phi'}(x)} = \nabla_{\phi'}^T\nabla_{\phi'} \log q_{\phi'}(x)$ the Hessian matrix. Then we can write Taylor expansion as follows <br />
$$ D_{KL}(q_\phi(x) || q_{\phi + \delta}(x)) = \frac{1}{2} \delta^T \mathbb{E}_{q_\phi(x)}\left[  -H_{\log q_{\phi'}(x)} \mid_{\phi' = \phi}\right] \delta + o(||\delta||^2) = \delta^T F_x(\phi) \delta + o(||\delta||^2)$$ <br />
which proofs the statement.
[/proof]

Within the Bernoulli example, we already observed a close relationship to the variance of the random variable. This relationship is not as universal as you would guess! At least within the exponential family, there is a clear connection to the covariance of the test statistic. Yet, recall that this is only with respect to the natural parameters. In general, we can relate these quantities through a lower bound: 

And here are some other nice properties
<dir class=theorem>
 The Fisher information satisfies the following properties <br />
  (i)  $F(\phi)$ is symmetric and positive semi-definite. <br />
  (ii) $F(\phi)$ is positive definite if the statistical model is identifiable (there cannot exist two parameters $\phi_i \neq \phi_j$ such that $p_{\phi_i} = p_{\phi_j}$)<br />
  (iii) If $[\phi]_i$ and $[\phi]_j$ are independent then $[F(\phi)]_{ij} = 0$ i.e. then $[\phi]_i$ and $[\phi]_j$ can be estimated seperatly <br />
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

An estimator is typically just a function $f : X^N \rightarrow \phi$, which maps a set of observations to a specific parameter. Yet in certain cases, it may be less practical to work with the raw data and we instead want to preprocess it. We may even use a statistic to summarize the data, i.e. $T: X^N \rightarrow \mathcal{T}$. This may simplify the design of an estimator significantly as we now only have to consider a single or a few statistics. Yet, do we lose information by just considering the statistic? Most information should always be within the raw data, right? This is indeed true (according to Fisher information) !

<div class="theorem">
 (Data-processing inequality):   Be $X$ an random variable and be $T(X)$ an arbitrary transformation independent of $\phi$. Then
 $$ F_{T(X)}(\phi) \leq F_X(\phi)$$
Where for matrices we use the standard Loewner order i.e. $A \leq B \iff A - B$ is positive semi-definite. The inequality is an equality if and only if $T$ is a sufficient statistic.
</div>
[proof]
As the Fisher information satisfies the additive chain rule it follows that for any random variable $Y$ we have that <br />
$$ F_{Y, X}(\phi) = F_{Y| X}(\phi) + F_{X}(\phi) \geq F_X(\phi)$$ <br />
where the inequality holds because $F_{Y| X}(\phi)$ is postive semi definite. Thus it follows that <br />
$$ F_{T(X)}(\phi) \leq F_{T(X), X}(\phi) = \underbrace{F_{T(X)|X}(\phi)}_{=0} + F_X(\phi) = F_X(\phi).$$ <br />
Here $F_{T(X)|X}(\phi) = 0$ as $T(X)$ is a deterministic transform of $X$ and thus independent of $\phi$ given $X$.
[/proof]

So that's nice that Fisher information does indeed follow our intuition. So let's start to do inference. The most common type to obtain an estimator $f$ of data is the maximum likelihood method. We typically propose some family of parametric models $\mathcal{F} =\{ q_\phi \mid \phi \in \Phi \}$, given some data $x_1, \cdots, x_n \sim p$ we then select 

$$ \hat{\phi} = \arg\max_{\phi \in \Phi} \sum_{i=1}^n \log q_\phi(x_i)$$

Note as $X_1, \cdots, X_n$ are independent realization of random variables following measure $p$. The estimate $\hat{\phi}$ is thus itself a random variable. From major interest are the moments of this random variable, which are used to define two important measures of quality:
* **Bias**: A well-behaved estimator should at least be in expectation correct. This is quantified by the bias
  $$ b(\hat{\phi}) = \mathbb{E}[\hat{\phi}] - \phi^* $$
  where $\phi^*$ denotes the "true" (or best achievable) parameter. A parameter with zero bias is called *unbiased*.
* **Variance**: A unbiased point estimate is nice, but inefficient if it has high variance i.e. if it is widely spread around the mean then by performing a single point estimate you will likely land far away from it and so from the true parameter. 

As introduced there is a close connection not only for the MLE estimator but for any point estimate. While it is generally very hard to compute the exact variance of an estimator. We can derive a lower bound the *Carmér Rao lower bound*, given by

<div class="theorem">
Be $\phi^* \in \Phi$ the parameters of a probability density $q_{\phi^*}$. Assume that the Fisher information matrix $F_x(\phi)$ exists. Be $g(X)$ an statistical estimator of $\phi^*$ and assume that the first moment $\psi(\phi^*) =\mathbb{E}_{q_{\phi^*}}[g(X)]$ exists. Then the covariance of the estimator satisfies 

$$ \Sigma_{q_{\phi^*}}(g(X)) \geq \nabla_{\phi^*}\psi(\phi^*) \left[ F(\phi^*) \right]^{-1} \nabla_{\phi^*}\psi(\phi^*)^T$$

If the bias $b(\phi^*)$ is known. Then we can also rewrite it as

$$ \Sigma_{q_{\phi^*}}(g(X)) \geq (1 + b(\phi^*)) \left[ F(\phi^*) \right]^{-1} (1 + b'(\phi^*))^T$$

If the estimator is unbiased, then 

$$ \Sigma_{q_{\phi^*}}(g(X)) \geq F(\phi^*)^{-1}$$

</div>
[proof]
test
[/proof]

This is nice because it lets us evaluate the efficiency of estimators. Hence we call an estimator *efficient* if the variance equals the Cramér Rao lower bound.

In general, the distribution of an arbitrary statistical estimator can become very complicated. Yet, to be able to run statistical tests or construct confidence intervals, we have to know the actual distribution. Even for the "nice" MLE, this can be generally hard but at least a good asymptotic results exist

<dir class="theorem">
(Asymptotic normality of the MLE): Be $X_1, \dots, X_n \sim q_{\phi^*}$ and be $\hat{\phi}$ the MLE of $\phi^*$. Then 

$$ \sqrt{n}(\hat{\phi} - \phi^*) \rightarrow \mathcal{N}(\phi; 0, F(\phi^*)^{-1}) $$
</dir>
[proof]
test
[/proof]


### Experimental design

 By Theorem 8, we can estimate the standard deviation of the MLE by $\sigma_F = \sqrt{(n F_x(\phi^*))^{-1}}$ whenever $n$ is large enough. Observe that the standard deviation decreases whenever the Fisher information or $n$ is large. In practise we cannot control the true value, but we can effect the number of trials we perform i.e. by collecting more data. So it may be interesting to know how much samples we need such that the MLE has a standard deviation below a certain threshold...

 Let's again consider an Bernoulli experiment. We want to estimate the probability $\phi^*$ using the MLE $\hat{\phi}$ but want to ensure that the standard deviation $\sigma_F \leq \epsilon$. It follows that we have to ensure that

 $$ \sqrt{(n F_x(\phi^*))^{-1}} \leq \epsilon \iff  n \geq \frac{1}{\epsilon^2}F_x(\phi^*)^{-1}$$

 Note that in practise we do not know the ture value $\phi^*$, thus we may solve this problem for the worst case scenario i.e

$$ n \geq \frac{1}{\epsilon^2} \max_\phi F_x(\phi)^{-1} =  \frac{1}{\epsilon^2} F_x(0.5)^{-1} = \frac{1}{(2\epsilon)^2}$$

As a result to achive a standard deviation of $\sigma_F \leq 0.1$ we need $n \geq 25$ samples.


### Testing hypothesis

This also paves the way to construct approximate confidence intervals for any MLE estimator of an arbitrary statistical model. This is simply because we know how to construct confidence intervals or sets for Normal distributions. For simplicity we constrain ourself here to the one dimensional case. Here a $99\%$ confidence interval is given by 
$$ \left( \hat{\phi} - 2.96 \sigma_F, \hat{\phi} + 2.96 \right) $$
meaning that the unknown true value $\phi^*$ lies inside this interval in 99 out of 100 independent MLE estiamtes. 

<p align="center">
<figure>
  <img src="../../../assets/bernoulli_confidence.png" />
  <img src="../../../assets/bernoulli_confidence2.png" />
  <figcaption> The figures show the sampling distribution of the MLE for different number of datapoints, as well as the constructed $99\%$ confidence intervals. </figcaption>
</figure>
</p>

# Constructing confidence intervals

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
