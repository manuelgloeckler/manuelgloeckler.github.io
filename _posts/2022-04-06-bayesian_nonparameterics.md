---
layout: article
title: An interactive guide through Bayesian Nonparameterics
key: A1
tags: Bayesian ML, Statistics, Math
cover: https://s8.gifyu.com/images/cover_bnp9c3d89a85f1d0055.gif
comment: true
---

Traditional parametric models using a fixed and finite number of parameters can suffer from data over-or under-fitting when there is a mismatch between the complexity of the model (often expressed in terms of the number of parameters), the complexity of the true data generating process, and the amount of available data. As a result, we often have to do *model selection* to choose the right model from an ensemble of possible models. Unfortunately, model selection is an operation that is complicated and tedious, independently of the use of frequentist cross-validation or Bayesian marginal probabilities as the basis for selection.

The *Bayesian nonparametric* approach is an alternative to parametric modeling and selection, by adapting its complexity to the amount of available data or the complexity of the data generating process. The thereby typically unbound complexity mitigates underfitting, while the Bayesian approach for computing full posteriors mitigates overfitting. Note we refer to a model as parametric if it has a finite number of parameters. A nonparametric model in contrast assumes a priori an infinite number of parameters.

A Bayesian nonparametric (BNP) model defines a probability distribution over infinite-dimensional parameter space. This sounds complicated, yet as we will see in practice a BNP uses only a finite subset of the potentially infinite parameters to explain any finite set of observations.

## A parametric model: Mixture models

Let's start with a simple parametric model that is often used for clustering or density estimation purposes: The Gaussian Mixture model.

We can write down the generative model for $K$ clusters as follows:

$$ \mu_k \sim \mathcal{N}(\mu_0, \Sigma_0)
$$

$$ \pi \sim Dir(\alpha)$$

$$ z_n \sim Cat(\pi)$$

$$ x_n \sim \mathcal{\mu_{z_n},\Sigma}$$

Thereby we call $\mu_k$ the cluster means, one for each of the $K$ clusters. On the other hand, $z_n$ represents the cluster memberships and $\pi$ the mixing coefficients.

For any finite set of $N$ observations $x_n \in \mathbb{R}^d$, the model has $N + Kd + K - 1$ parameters i.e. the cluster memberships, the cluster means, and the mixing coefficients. Thus we indeed can call it *parametric*. Notice that we still have other *hyperparameters* i.e. the parameters of the priors $\mu_0, \Sigma_0, \alpha, \Sigma$. Let's make it easy, we choose $\mu_0 = 0$ and $\Sigma_0 = I$, this makes a lot of sense if we standardize the data. Further let's say $\Sigma=0.1I$. On the other hand, finding a good choice of $\alpha$ is harder. What does $\alpha$ change? In the plot below you can try, try to answer this question yourself!

{% include finite_mixture_model.html %}

As present within the generative model, $\alpha$ determines the mixing coefficients $\pi$. As $\alpha \rightarrow 0$ the Dirichlet distribution concentrates at the corners of the unit simplex i.e. we only have a few dominant clusters and many very small. As $\alpha \rightarrow \infty$ the distribution converges to a point mass at $1/K$.

Now for any dataset $X$ we observe, we can infer the parameters just by using Bayes' theorem:

$$ p(Z, \pi, \mu_{1:K}|X) \propto p(X|\mu_{1:K},Z)p(Z|\pi)p(\pi)p(\mu_{1:K}) = \prod_{n=1}^N \mathcal{N}(x_n; \mu_{z_n}, 0.1I)Cat(z_n;\pi)Dir(\pi;\alpha) \prod_{k=1}^K \mathcal{N}(\mu_k;0,I)$$

Whereas we cannot retain a close form solution, we can sample from it using MCMC-methods e.g. Gibbs Sampling.

Unfortunately, this model has one limitation. If the data requires more than $K$ cluster centers, then we have a problem. Hence we have to do *model selection* i.e. to we have to choose the best $K$. In the end, we shouldn't need more clusters than data points $N$ we observe, right? That's not always true e.g. in many cases we just did sample from some of the latent clusters yet. If we would collect more and more data we would observe more and more clusters showing up. When there are 1000 latent components and we observe only 100 data points then we observe less than 100 clusters, right?

So let's look what will happen if we sequentially observe one datapoint $x_i$ at a time. There are two events that can happen:

* $x_i$ can be part of a cluster we already observed.
* $x_i$ can be part of a **new** cluster.

Let's say we have $K=1000$ components. How many points do we need to observe all 1000 components? At which rate do we observe new components? As always try to answer the question yourself, the animation below will help by simulation this process.

{% include rate_of_clusters.html %}

Thanks to this letting $K \rightarrow \infty$ does not seem as crazy as before, right? In the end, we will observe a finite number of observations and thus only a finite number of components! Note that the simulation above uses $K = 1000$ so in the end, it will become a flat line as we approach 1000 (if you wait forever :D). In the next section, we will discuss how we can set $K = \infty$.

## The Stickbreaking construction

The main problem we will encounter as we let $K \rightarrow \infty$ is that the Dirichlet distribution becomes ill-defined. In the end, how should we even sample a vector of infinite length? At a certain point at least your memory will say goodbye!

As always the solution is to be lazy! After a thousand data points we only observed 200 clusters in the above simulation, so why should we sample mixture coefficients for all other components, when we can just lazily generate a new one only if we need it. Let's assume $K=2$, an alternative way to sample from a Dirichlet is to use the Beta distribution i.e.

$$ \pi_1 = \mathcal{B}(\alpha_1, \alpha_2) \quad \text{ and } \quad  \pi_2 = 1-\pi_1$$

It is rather easy to see that $(\pi_1, \pi_2) \sim Dir((\alpha_1,\alpha_2))$ (by just seeing that the marginal distribution of a Dirichlet is always a Beta). So let's generalize this to an arbitrary $K$ which is known as *Stickbreaking*. 

Assume we have a stick of length $1$ (representing all the probability mass). Our goal is to break the strick into $K$ pieces, by definition these pieces must thus sum to one.  We start as following:

* Draw $\phi_1 \sim \mathcal{B}(\alpha_1, \sum_{k=2}^K \alpha_k)$. Set $\pi_1 = \phi_1$ and break a part of lenght $\pi$ from the stick.
* Draw $\phi_2 \sim \mathcal{B}(\alpha_2, \sum_{k=3}^K \alpha_k)$. The remaining stick has length $(1-\pi_1)$, thus break of a part of length $\pi_2 = \phi_2(1-\pi_1)$.
* ...
* Draw $\phi_i \sim \mathcal{B}\left(\alpha_i, \sum_{k=i+1}^K \alpha_k\right)$. Set $\pi_i =  \prod_{j=1}^{K-1}(1-\phi_j) \phi_i = \phi_i \cdot \left(1-\sum_{j=1}^{i-1}\pi_i\right)$
* ...
* Set $\pi_K = 1- \sum_{k=1}^{K-1}\pi_k$.

So by this way, we can lazily generate a sequence of $K$ numbers that sum to one. But why did we stop at $K$ pieces?  Can't we just continue forever?

This idea turns out to be (almost) true, we indeed can generalize the procedure above:

* Draw $\phi_1 \sim \mathcal{B}(a_1, b_1)$. Set $\pi_1 = \phi_1$.
* Draw $\phi_2 \sim \mathcal{B}(a_2, b_2)$. Set $\pi_2 = (1-\phi_1)\phi_2
* ...
* Draw $\phi_i \sim \mathcal{B}(a_i,b_i)$. Set $\pi_i =  \prod_{j=1}^{K-1}(1-\phi_j) \phi_i$
* ...

By this way we generate a infinite sequence $\pi_1, \pi_2, \cdots$ from which we at least can guarantee that $\sum_{k=1}^\infty \pi_k \leq 1$. But wait, why does it not equal one?

This can be easily seen with an counter-example: Consider the sequence $\pi_i = 1/(2^n+2)$, then $\sum_{i=1}^\infty \pi_i = 1/2 \leq 1$. Thus we have to make sure that we cannot sample such sequences, or at least that the probability to sample them is zero (the set of such sequences should have zero measure). As it turns out that we actually can achieve this by just restricting the set of parameters $a_i, b_i$ within the process (see Ishwaran, James 2001).

As it turns out if we choose $a_1=1$ and $\beta = \alpha > 0$, then with probability 1 all sequence we sample will converge to 1! That's perfect and all we need, we call the underlying process now **Dirichlet process stick-breaking**, the underlying distribution is called the **GEM** distribution. We again can look at the number of observed "clusters", if you give the below animation infinite time you will see that indeed it is going to infinity. In contrast to the previous plot, however, the only thing which will stop you here is your finite time and memory...

{% include rate_of_clusters_infinite_DP.html %}

## Dirichlet process mixture models

After discovering the GEM distribution, we can now easily write up a generative model for a mixture model with an infinite number of components.

$$ \pi = (\pi_1, \pi_2, \dots) \sim GEM(\alpha)$$

$$ \mu_k \sim \mathcal{N}(\mu_0, \Sigma_0) \text{ for } k=1,2, \dots $$

$$ z_n \sim Cat(\pi)$$

$$ x_n \sim \mathcal{N}(\mu_{z_n}, \Sigma)$$

So let's look at some samples from this process. Note that we actually cannot sample it completely, as this would require infinite time and memory. But we can generate data sequentially as before! In the below animation you can test this, especially you can experiment with the one hyperparameter $\alpha$ and its influence on the number of clusters we actually would observe in a finite dataset. Note that all of the samples have in principle an infinite number of components, yet in finite time (and with finite memory) you won't be able to actually observe all of them ;)

{% include dp_mixture.html %}

## Dirichlet process: A bit more formal

Be $\Theta$ be a parameter space, be $G_0$ a base measure over $\Theta$. Again we draw $\pi \sim GEM(\alpha)$ and $\theta_k \sim G_0$. Then we call $G = \sum_{k=1}^\infty \pi_k \delta_{\theta_k}$ a draw from a **Dirichlet process (DP)** $G \sim DP(\alpha, G_0)$. Here we denote with $\delta_{\theta_k}$ a indicator function on a specific $\theta_k$ sampled from the base measure. Thus every draw from a DP will be this infinite object! We can recover the previous scenario by just choosing $\Theta = \mathbb{R}^2$ and $G_0 = \mathcal{N}(\mu_0, \Sigma_0)$. Thus all the $\theta_k$ here now correspond to the infinite number of possible cluster centers, each associated with a specific mixing coefficient $\pi_k$.

In fact, these objects $G$ have a specific property. The are measures over the parameter space $\Theta$ (in fact even a probability measure!). To see that consider some set $A \subset \Theta$ then
$$ G(A) = \sum_{k=1}^\infty \pi_k \delta_{\theta_k}(A) = \sum_{k: \theta_k \in A} \pi_k$$
So in fact the DP is a distribution over **random measures**.

So that's great, but wait we call it a *Dirichlet* *process*. So, it should be a *stochastic process* i.e. an indexed collection of random variables. And it should have something in common with Dirichlet distribution, right? So what are our random variables, are they Dirichlet distributed? And what is our index set?

So let's consider some fixed set $A \subset \Theta$. Then $G(A)$ is random, right? Simply because $G$ is random. In fact by construction we can easily see that $G(A) \in [0,1]$. This looks typically like a marginal sample drawn from a Dirichlet distribution, right? So let's try to complete it. If we consider $B = \Theta \backslash A$, it's easy to see that by construction $G(A) + G(B) = 1$. So let's consider the random vector $(G(A), G(B))$. We know that any realization must sum to one, so it really looks like a draw from Dirichlet distribution. But what are the parameters? Intuitively it must be proportional to the volume of $A$ under the base measure $G_0$, as if $A$ covers most of the support of $G_0$ then most of the indicator functions must be within it. And yes that's exactly right in fact it turns out that $(G(A), G(B)) \sim Dirichlet((\alpha G_0(A), \alpha G_0(B)))$ ([Here the details](http://www.people.vcu.edu/~dbandyop/pubh8472/StickBreaking.pdf)). So indeed we found that a partition $A,B$ of $\Theta$, will generate a random vector distributed according to a Dirichlet distribution. In fact, this will hold for any finite partition of $\Theta$! And hence we found the index set of the stochastic process, these are all the different partitions that exist. All this now allows us to formally define this process:

<dir class="definition">
 (Dirichlet Process): A Dirichlet Process is a distribution of a random probability measure $G$ over a measurable space $(\Theta, \sigma(\Theta))$, such that for any finite partition $(A_1, \dots, A_K)$ of $\Theta$, we have

 $$ (G(A_1), \dots, G(A_K)) \sim Dir(\alpha G_0(A_1), \dots, \alpha G_0(A_r)) $$

 Here $G(A_i) = \int_{A_i} dG$ and $G_0(A_i) = \int_{A_i}dG_0$*
</dir>

We can summarize, that indeed the DP is a stochastic process that defines a distribution over distributions (probability measures) i.e. each draw from a DP is itself a distribution. It is called the Dirichlet process because it has Dirichlet distributed finite-dimensional marginal distributions, just like the Gaussian process, another popular stochastic process used from Bayesian nonparametric regression. Distribution drawn from a DP is **discrete**, but infinite!

As consequence, an equivalent way to write down the Dirichlet process mixture model is:

$$ G \sim DP(\alpha, \mathcal{N}(\mu_0, \Sigma_0))$$

$$ \mu_k \sim G$$

$$ x_n \sim \mathcal{N}(\mu_k, \Sigma)$$

Note that we no longer require to use the latent variables $z_n$ i.e. the cluster memberships.

## The Dirichlet process posterior

Let $G \sim DP(\alpha, G_0)$. Since $G$ is itself a (random) distribution, we can in draw samples $\theta_1, \dots, \theta_n \sim G$. Note that $\theta_i's$ take values in $\Theta$ since $G$ is a distribution over $\Theta$. Let's assume we are interested in the posterior distribution of $G$ given that we observe $\theta_1, \dots, \theta_n$.

Let $A_1, \dots, A_r$ be a finite measurable partition of $\Theta$. Be $n_k = \sum_{i=1}^n I(\theta_i \in A_k)$ the number of observed values in $A_k$. Then by the conjugacy between the Dirichlet and multinomial distribution, we have
$$ (G(A_1,), \dots, G(A_r)) | \theta_1, \dots, \theta_n \sim Dir(\alpha G_0(A_1) + n_1, \dots, \alpha G_0(A_r) + n_r)$$
Since the above is ture for all finite measurale paritions, the posterior distribution over G must be a DP as well (as all marginals are still Dirichlet). So how do we have to update the parameters? As it turns out we only have to investigate our definition:

<dir class="definition">
 (Dirichlet Process Posterior): Be $G \sim DP(\alpha, G_0)$*. Be $\theta_1, \dots, \theta_n \sim G$, then

 $$ G|\theta_1, \dots, \theta_N \sim DP(\alpha + N, \hat{G_0}) $$
 
 Here $\hat{G_0} = \frac{\alpha G_0 + \sum_{i=1}^N \delta_{\theta_i}}{\alpha + N}$
</dir>
You can verify easily that for any finite partition, we obtain a Dirichlet posterior distribution as derived above.

Notice that we can rewrite the posterior as following
$$ G|\theta_1, \dots,\theta_n \sim DP\left(\alpha + N, \frac{\alpha}{\alpha + N} G_0 + \frac{N}{\alpha + N} \frac{\sum_{i=1}^N \delta_{\theta_i}}{n}\right)$$
Thus the posterior is a weighted combination of the base distribution $G_0$ and the empirical distribution. The weight associated with the base distribution is proportional to $\alpha$, while the empirical distribution has a weight proportional to the number of observations $N$. Thus we can interpret $\alpha$ as the mass associated with the prior, taking $\alpha \rightarrow 0$ will render the prior non-informative in the sense that the predictive distribution is just given by the empirical distribution.

You can play with the posterior distribution in the below animation. You can add observations by just taping into the figure with your mouse (a red dot will appear!). At this moment you condition our previous prior on this particular observation. You can look at multiple samples for different alpha and verify the above intuition we build up.

{% include dp_mixture_posterior.html %}

Note: We draw the means from the GP and within the above Dirichlet Mixture simulation we do condition the means $\mu$, not the observation $x$. We will come to the second case latter.

## The predictive distribution and the Chinese restaurant process

Consider again $G \sim DP(\alpha, G_0)$ and drawing an i.i.d. sequence $\theta_1, \theta_2, \dots \sim G$. Then consider the predicitve distribution for $\theta_{n+1}$, conditioned on $\theta_1, \dots, \theta_n$ with $G$ marginalized out.

Since $\theta_{n+1}| G, \theta_1, \dots, \theta_n \sim G$, as $\theta_{n+1}$ is conditionally independent of $\theta_1, \dots, \theta_n$ given $G$. We have that
$$ P(\theta_{n+1} \in A |\theta_1, \dots, \theta_n) = \mathbb{E}\left[ G(A) | \theta_1, \dots, \theta_N \right] = \frac{1}{\alpha + N} \left( \alpha G_0(A) + \sum_{i=1}^N \delta_{\theta_i}(A) \right)$$
Thus with $G$ marginalized out
$$ \theta_{N+1}|\theta_1, \dots, \theta_N \sim \frac{\alpha G_0 + \sum_{i=1}^N \delta_{\theta_i}}{\alpha + N}$$

Thus the posterior base measure is also the predictive distribution, which indeed allows us to draw samples as follows:

* With probability $\frac{\alpha}{\alpha + N}$ draw $\theta_{N+1} \sim G_0$
* Else draw some $\theta_1, ..., \theta_N$ uniformly.

The Chinese Restaurant Process describes this generative model using a metaphor. Customers enter a Chinese restaurant with an infinite amount of tables. The first customer enters the restaurant and sits at the first table. The second customer enters the restaurant and decides either to sit with the first customer or by himself, by opening a new table. In general, the $n+1$ customer can either sit at one of the $K$ opened tables or create a new one with probability proportional to the number of customers already sitting there or open a new table with probability proportional to the concentration parameter $\alpha$.

Note that the previous simulation actually exactly follows this process.

## Applications

In this section we finally come to some usefull applications. 

### Dirichlet Process Mixture posterior

Above we already 

{% include trinket-open %}
# Loading tweet data
{% include trinket-close %}


### Simulation-based inference

Whereas the last example (an most of this post) got an fully Bayesian treatment we for now will switch to the dark side. 



