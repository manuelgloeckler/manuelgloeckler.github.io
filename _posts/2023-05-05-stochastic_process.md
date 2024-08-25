---
layout: article
title: Stochastic Processes
tags: Statistics, Math, Probability Theory
aside:
    toc: true
cover: https://2.bp.blogspot.com/-VJwBBmGlQ9w/XalANlu2nDI/AAAAAAAAWRo/i3L26OxdcD8wCQy5GPzCLBH-Ix7Lq014ACLcBGAsYHQ/s1600/ezgif.com-crop%2B%25285%2529.gif
comment: true
---

In this blog post, we will delve into the fascinating realm of stochastic processes. We'll begin with a concise yet informative introduction to stochastic processes. Subsequently, we'll explore various properties, introduce crucial examples, and investigate their characteristics.

## Introduction

First let us recall the definition of a random variable. A random variable is a random number
appearing as a result of a random experiment. If the random experiment is modeled by a
probability space $(\Omega, \mathcal{F}, \mathbb{P})$, then a $\mathcal{X}$ valued random variable is defined as a measurable function $X: \Omega \rightarrow \mathcal{X}$. Performing the random experiment means choosing the outcome $\omega \in \Omega$ at random according to the probability measure $\mathbb{P}$. Then, $X(\omega)$ is the value of the random
variable which corresponds to the outcome $\omega$.


<dir class="definition">
<b>(Stochastic process):</b> Let $(\Omega, \mathcal{F}, \mathbb{P})$ a probability space and $T$ an arbitrary set (called the <i>index set</i>). Any collection of random variables 
$$ X = \{X_t: t \in T\}$$
is called a stochastic process.
</dir>

In other words, a stochastic process is a "random function". Classically a function returns a specific value if provided with an input $t$. In contrast, a stochastic process returns a random value i.e. a realization of the random variable if provided with an input $t$. Notably if we fix the outcome of the random experiment we indeed end up with a classical deterministic function.

<dir class="definition">
<b>(Sample path, Realization, Trajectory):</b> Given a specif outcome $\omega \in \Omega$, the function $t \rightarrow X_t(\omega)$ is called a sample path, realization or trajectory of the stochastic process $X$.
</dir>
Traditionally the index set $T$ is refered to as *time*. We typically distinguish
- **Discrete-time stochastic processes**: $T = \mathbb{Z}$ or $T = \mathbb{N}$. In this case, the stochastic process is a sequence of random variables.
- **Continuous-time stochastic processes**: $T = \mathbb{R}$. In this case, the stochastic process is a function of time.
- **Random fields**: $T = \mathbb{R}^d$. In this case, the stochastic process is a function of space.

Note that any stochastic process with a finite index set is just a random vector. Hence, we will focus on stochastic processes with an infinite index set.

### Discrete-time stochastic processes

We look at the following examples of discrete-time stochastic processes:
1. **Iid. noise:** Let $X =\{X_n : n \in \mathbb{N}\}$ with $X_n$ independent and identically (i.i.d.) random variables. Then, the stochastic process $X$ is called iid. noise. The underling probability space is $\Omega = \mathcal{X}^\mathbb{N}$ i.e. the set of all sequences. The probability measure is the product measure of the probability measure of the random variable $X_n$ on the product $\sigma$-Algebra.
2. **Random walk:** Let $X$ be a iid. noise process, then $ S_n = X_1 + \dots + X_n$ is called a random walk. The probability space can be constructed as above. 
3. **Geometric random walk:** Let $X$ be a iid. noise process, then $$G_n = X_1 \cdot \dots \cdot X_n$$ is called a geometric random walk.

In the interactive plot below, you can see the sample paths of the above stochastic processes.

{% include discrete_stochastic_process.html %}

For instance, the geometric random walk exhibits significant volatility, with the potential to grow rapidly and then suddenly drop to zero.  In contrast, the random walk is more stable, experiencing growth at a slower pace and never dropping to zero. Changing the underlying distribution from Normal to Cauchy transforms relatively smooth sample paths into highly discontinuous ones. So, by simply altering the underlying distribution, we obtain vastly different sample paths.



### Continuous-time stochastic processes

Now, let's delve into examples of continuous-time stochastic processes:

  1. **Lévy process**: Lévy processes are types of stochastic processes that can be considered as generalizations of random walks in continuous time. A stochastic process $X$ is a Lévy process if for $0 \leq t_1 \leq \dots \leq t_n$ the corresponding increments $X_{t_2} - X_{t_1}, \dots, X_{t_n} - X_{m-1}$ are independent and the distribution of them does only depend on the difference $\mid t_i - t_{i-1} \mid$.  Many important stochastic processes are Lévy processes, e.g. Wiener process,Cauchy process, Poisson process, etc.
     1. **Wiener process**: Assume $W_0 = 0$, then the Wiener process is a stochastic process $W = \{W_t : t \geq 0\}$    with the following properties:
        - (i) $W_t$ has independent increments.
        - (ii) $W_t - W_s \sim \mathcal{N}(0,\mid t-s \mid )$ for all $t,s > 0.
     2. **Cauchy process**: Assume $C_0 = 0$, then the Cauchy process is a stochastic process $C = \{C_t : t \geq 0\}$ with the following properties:
        - (i) $C_t$ has independent increments.
        - (ii) $C_t - C_s \sim \mathcal{C}(0,\mid t-s \mid )$ for all $t,s > 0$.
     3. **Poisson process**: Assume $P_0 = 0$, then the Poisson process is a stochastic process $P = \{P_t : t \geq 0\}$ with the following properties:
        - (i) $P_t$ has independent increments.
        - (ii) $P_t - P_s \sim \mathcal{P}(\lambda \mid t-s \mid )$ for all $t,s > 0$.

  2. **Random polynomials:** Let $\xi_0, \dots, \xi_d$ be i.i.d. random variables. Then, 
   $$P_n(t) = \sum\limits_{k=0}^d \xi_k t^k$$ 
   is called a random polynomial. The event space $\Omega = \mathbb{R}^d$ with classical Borel $\sigma$-Algebra and an arbitrary probability measure $\mathbb{P}$.

{% include continuous_stochastic_process.html %}

So let's define some properties that these processes can have and investigate them on some of these examples.

## Properties

We will now introduce some properties that a stochastic process can have. We will also investigate some of these properties on the examples we have seen above.

### Finite-dimensional distributions

Even if we have an infinite index set, we can still just look at a finite number of indices. This can be thought of projecting the stochastic process onto a finite-dimensional subspace. The resulting distribution is called the finite-dimensional distribution.

<dir class="definition"><b>Finite-dimensional distributions</b>
The finite-dimensional distributions of a stochastic process $X$ are the joint distributions of the random variables $X_{t_1}, \dots, X_{t_n}$ for all $t_1, \dots, t_n \in T$. We denote the finite-dimensional distributions by $P_{t_1, \dots, t_n}$.
</dir>

We can consider the collection of all finite-dimensional distributions of $X$, denoted as $\mathcal{P}$:

$$\mathcal{P} := \{ P_{t_1, \ldots, t_n} : n \in \mathbb{N}, t_1, \ldots, t_n \in T \}.$$

Now the question arises: Can we associate a stochastic process with any collection of finite-dimensional distributions? Clearly this question only makes sense if the collection satisfy some properties:

   1. **Permutation Invariance:** For any permutation $\pi: \{1, \ldots, n\} \rightarrow \{1, \ldots, n\}$, the following holds:
      For all $n \in \mathbb{N},$ for all $t_1, \ldots, t_n \in T,$ and for all $B_1, \ldots, B_n \in \mathcal{B}(\mathbb{R}),$

   $$P_{t_1, \ldots, t_n}(B_1 \times \ldots \times B_n) = P_{t_{\pi(1)}, \ldots, t_{\pi(n)}}(B_{\pi(1)} \times \ldots \times B_{\pi(n)}).$$

   2. **Projection Invariance:** For all $n \in \mathbb{N},$ all $t_1, \ldots, t_n, t_{n+1} \in T,$ and all $B_1, \ldots, B_n \in \mathcal{B}(\mathbb{R}),$ the following holds:

   $$P_{t_1, \ldots, t_n, t_{n+1}}(B_1 \times \ldots \times B_n \times \mathbb{R}) = P_{t_1, \ldots, t_n}(B_1 \times \ldots \times B_n).$$

This just basically assumes that the set is consistent with respect to permutations and projections, exactly the "operations" we performed defining the finite-dimensional distributions of a stochastic process. 

Then indeed the answer is yes, and this is precisely what Kolmogorov's Existence Theorem addresses.

<div class="theorem"><b>(Kolmogorov's Existence Theorem)</b> 
Given a non-empty set $T$ and a collection of probability measures $\mathcal{P} = \{P_{t_1, \ldots, t_n} : n \in \mathbb{N}, t_1, \ldots, t_n \in T\}$ with the properties of permutation invariance and projection invariance as stated above, there exists a probability space $(\Omega, \mathcal{F}, P)$ and a stochastic process $\{X_t : t \in T\}$ on this probability space. Importantly, this stochastic process has finite-dimensional distributions matching the collection $P$. In other words, for every $n \in \mathbb{N}$ and every $t_1, \ldots, t_n \in T$, the distribution of the random vector $(X_{t_1}, \ldots, X_{t_n})$ coincides with $P_{t_1, \ldots, t_n}$.
</div>
<details>
<summary><i>Proof.</i></summary>
<div class="proof">
<br>
<b>Step 1:</b> Let us construct $\Omega$ first. Usually, $\Omega$ is the set of all possible outcomes of some random experiment. In our case, we would like the outcomes of our experiment to be functions (the realizations of our stochastic process). Hence, let us define $\Omega$ to be the set of all functions defined on $T$ and taking values in $\mathbb{R}$:

$$\Omega = \mathbb{R}^T = \{f : T \rightarrow \mathbb{R}\}.$$

<b>Step 2:</b> Let us construct the functions $X_t : \Omega \rightarrow \mathbb{R}$. We want the sample path $t \mapsto X_t(f)$ of our stochastic process corresponding to an outcome $f \in \Omega$ to coincide with the function $f$. In order to fulfill this requirement, we need to define:

$$X_t(f) = f(t), \quad f \in \mathbb{R}^T.$$

The functions $X_t$ are called the canonical coordinate mappings. For example, if $T = \{1, \ldots, n\}$ is a finite set of $n$ elements, then $\mathbb{R}^T$ can be identified with $\mathbb{R}^n = \{f = (f_1, \ldots, f_n) : f_i \in \mathbb{R}\}$. Then, the mappings defined above are just the maps $X_1, \ldots, X_n : \mathbb{R}^n \rightarrow \mathbb{R}$ which map a vector to its coordinates:

$$X_1(f) = f_1, \ldots, X_n(f) = f_n, \quad f = (f_1, \ldots, f_n) \in \mathbb{R}^n.$$

<b>Step 3:</b> Let us construct the $\sigma$-algebra $\mathcal{F}$. We have to define what subsets of $\Omega = \mathbb{R}^T$ should be considered as measurable. We want the coordinate mappings $X_t : \Omega \rightarrow \mathbb{R}$ to be measurable. This means that for every $t \in T$ and every Borel set $B \in \mathcal{B}(\mathbb{R})$, the preimage

$$X_t^{-1}(B) = \{f : T \rightarrow \mathbb{R} : f(t) \in B\} \subset \Omega$$

should be measurable. By taking finite intersections of these preimages, we obtain the so-called cylinder sets, that is sets of the form:

$$A_{B_1, \ldots, B_n}^{t_1, \ldots, t_n} := \{f \in \Omega : f(t_1) \in B_1, \ldots, f(t_n) \in B_n\},$$

where $t_1, \ldots, t_n \in T$ and $B_1, \ldots, B_n \in \mathcal{B}(\mathbb{R})$. If we want the coordinate mappings $X_t$ to be measurable, then we must declare the cylinder sets to be measurable. Cylinder sets do not form a $\sigma$-algebra (just a semi-ring).

This is why we define $\mathcal{F}$ as the $\sigma$-algebra generated by the collection of cylinder sets:

$$\mathcal{F} = \sigma \left\{ A_{B_1, \ldots, B_n}^{t_1, \ldots, t_n} : n \in \mathbb{N}, t_1, \ldots, t_n \in T, B_1, \ldots, B_n \in \mathcal{B}(\mathbb{R}) \right\}.$$

We will call $\mathcal{F}$ the cylinder $\sigma$-algebra. Equivalently, one could define $\mathcal{F}$ as the smallest $\sigma$-algebra on $\Omega$ which makes the coordinate mappings $X_t : \Omega \rightarrow \mathbb{R}$ measurable.

Sometimes cylinder sets are defined as sets of the form:

$$A_{B}^{t_1, \ldots, t_n} := \{f \in \Omega : (f(t_1), \ldots, f(t_n)) \in B\},$$

where $t_1, \ldots, t_n \in T$ and $B \in \mathcal{B}(\mathbb{R}^n)$. One can show that the $\sigma$-algebra generated by these sets coincides with $\mathcal{F}$. <br>

<b>Step 4:</b> We define a probability measure $P$ on $(\Omega, \mathcal{F})$. We want the distribution of the random vector $(X_{t_1}, \ldots, X_{t_n})$ to coincide with the given probability measure $P_{t_1, \ldots, t_n}$, for all $t_1, \ldots, t_n \in T$. Equivalently, we want the probability of the event $\{X_{t_1} \in B_1, \ldots, X_{t_n} \in B_n\}$ to be equal to $P_{t_1, \ldots, t_n}(B_1 \times \ldots \times B_n)$, for every $t_1, \ldots, t_n \in T$ and $B_1, \ldots, B_n \in \mathcal{B}(\mathbb{R})$.

However, with our definition of $X_t$ as coordinate mappings, we have:

$$\{X_{t_1} \in B_1, \ldots, X_{t_n} \in B_n\} = \{f \in \Omega : X_{t_1}(f) \in B_1, \ldots, X_{t_n}(f) \in B_n\} = \{f \in \Omega : f(t_1) \in B_1, \ldots, f(t_n) \in B_n\} = A_{B_1, \ldots, B_n}^{t_1, \ldots, t_n}.$$

Hence, we must define the probability of a cylinder set $A_{B_1, \ldots, B_n}^{t_1, \ldots, t_n}$ as follows:

$$P[A_{B_1, \ldots, B_n}^{t_1, \ldots, t_n}] = P_{t_1, \ldots, t_n}(B_1 \times \ldots \times B_n).$$

It can be shown that $P$ can be extended to a well-defined probability measure on $(\Omega, \mathcal{F})$. This part of the proof is non-trivial but similar to the extension of the Lebesgue measure from the semi-ring of all rectangles to the Borel $\sigma$-algebra. We will omit this argument here. The properties of permutation invariance and projection invariance are used to show that $P$ is well-defined. 
</div>
</details>

### Mean and Covariance

As mean and variance are important for random variables, we also can define similar moments for a stochastic process. However, here we obtain mean and covariance functions that depend on time!

<dir class="definition"><b>Mean and covariance function</b>
If $X$ is a stochastic process with outcomes in $\mathbb{R}^d$ and an index set $\mathbb{T}$, then its mean and covariance functions, whenever they exist, are denoted by

$$ \mu(t) = \mathbb{E}[X_t] = \int x p_t(x)dx$$

$$ k(t_1,t_2) = Cov[X_{t_1},X_{t_2}] = \mathbb{E}[(X_{t_1}-\mu(t_1))(X_{t_2}-\mu(t_2))]$$

The covariance function is often also called a <b>kernel</b>.
</dir>

##### Example: Wiener process

The mean function, $$\mu(t)$$, represents the expected value of the process $$W_t$$ at time $$t$$. Since $$W_0 = 0$$, we know that $$\mu(0) = 0$$. For $$t > 0$$, we can compute $$\mu(t)$$ as follows:

$$
\mu(t) =\mathbb{E}[W_t] = \mathbb{E}[W_t - 0] = \mathbb{E}[W_t - W_0] = 0
$$

as any increment $W_t - W_0 \sim \mathcal{N}(0, t)$ has mean zero.


The covariance function, $$k(t_1, t_2)$$, can be computed as follows. Assuming $t_1 < t_2$, we have:

$$
k(t_1, t_2) = \mathbb{E}[(W_{t_1} - \mu(t_1))(W_{t_2} - \mu(t_2))] = \mathbb{E}[W_{t_1} W_{t_2}]
$$

Now substituting $W_{t_2} = W_{t_1} + (W_{t_2} - W_{t_1})$ we obtain: 

$$ k(t_1, t_2) = \mathbb{E}[W_{t_1} (W_{t_1} + (W_{t_2} - W_{t_1}))] = \mathbb{E}[W_{t_1}^2] + \mathbb{E}[(W_{t_1} - W_{t_0}) (W_{t_2} - W_{t_1})] =  \mathbb{E}[(W_{t_1} - W_{t_0})^2] = t_1$$

As $W_{t_1} - W_{t_0} \sim \mathcal{N}(0, t_1)$. Equivalently, for $t_1 > t_2$ we obtain $k(t_1, t_2) = t_2$ and thus we can write the covariance function as follows:
$$
k(t_1, t_2) = \min(t_1, t_2)
$$

### Stationarity and equilibrium
The time domain is rather large; in fact, it is uncountably large. Thus, it is hard to analyze the global behavior. However, as the next result shows, there is one property for which the process will behave "similarly" over the whole time domain.

<dir class="definition"><b>Strictly stationary process</b>
A stochastic process $X_t$ with an index set $\mathbb{T}$ is said to be strictly stationary if the finite-dimensional distributions are invariant to translation. That is, for every finite set $t_{1:N} = \{t_1, \dots, t_N\} \subset \mathbb{T}$ and for every $\tau \in \mathbb{T}$, the following holds:
$$X_{t_1 + \tau} , X_{t_2 + \tau}, \dots, X_{t_N + \tau}  =_{p} X_{t_1} , X_{t_2}, \dots, X_{t_N}$$
That is, the joint distributions of $X_{t_1 + \tau} , \dots, X_{t_N + \tau}$ and $X_{t_1} , \dots, X_{t_N}$ are the same.
</dir>

This globally "nice" behavior can also be verified by investigating the mean and covariance function. Notice that for all $\tau \in \mathbb{T}$, it holds for strictly stationary processes that 

$$ \mu(t + \tau) = \int x p_{\tau + t}(x)dx = \int  x p_t(x)dx = \mu(t) \Rightarrow \mu(t) = \mu$$

for some $\mu \in \mathcal{X}$. Hence the mean function is constant over the time domain! Further, we can rewrite the covariance function as

$$ k(t_1 + \tau, t_2 + \tau) = \int x_1x_2^Tp_{t_1 + \tau, t_2 + \tau} (x_1,x_2)dx_1dx_2 - \mu(t_1 + \tau)\mu^T(t_2 + \tau) $$

$$= \int x_1x_2^T p_{t_1,t_2}(x_1, x_2)dx_1dx_2 - \mu(t_1)\mu(t_2) = k(t_1, t_2)$$

Hence the covariance is invariant to translations. It only depends on the difference between time points $\mid t_2-t_1 \mid$, thus we may also write $k(\mid t_2-t_1 \mid )$.

The Wiener process is not strictly stationary. The only strictly stationary stochastic process are the iid. noise processes, by definition. 


<dir class="definition"><b>Asymptotic/Equilibrium distribution</b>
A stochastic process $X_t$ has an asymptotic or equilibrium distribution if there exists a distribution $\pi$ such that 
$$ \lim\limits_{t \rightarrow \infty} \mathbb{P}(X_t \in A) = \pi(A)$$
for all $A \in \mathcal{X}$.
</dir>

This simply means that over time the stochastic process converges to a certain distribution $\pi$. In fact if we now initialize the stochastic process $X_0 \sim \pi$, then this stochastic process is strictly stationary! Geometric random walks with Poisson distributed increments are an example of a stochastic process with an equilibrium distribution. This equilibrium distribution is a point mass at zero, as there is a non-zero probability that the process will eventually hit zero at some point in time, from which it will never be able to recover (as something multiplied with zero will always be zero...).

### Continuity and Differentiability

We already know the concepts of continuity and differentiability for functions. As we have seen, a stochastic process can be thought of as a function. Yet, classical definitions do not work for a stochastic process. Hence we have to define new concepts of continuity and differentiability. There are several ways to do this; we will introduce here the most straightforward. The theory of stochastic calculus is a whole topic on its own and we will not go into detail here.

The easiest way to define continuity and differentiability is to simply check if all the sample paths of the stochastic process are continuous or differentiable. This however is not very practical, as it is a very strict requirement. From the processes we have seen above, only the Polynomial process is continuous and differentiable almost surely.

Hence several weaker notions of continuity and differentiability have been introduced. We will introduce the mean square continuity and differentiability here.

<dir class="definition"><b>(Mean square continuous)</b>
A stochastic process $X_t$ is said to be mean square continuous at $t$ if 
$$ \lim\limits_{s \rightarrow t} \mathbb{E}[|X_s-X_t|^2] = 0$$
If $x$ is mean square continuous at each $t \in \mathbb{T}$, then $X_t$ is said to be mean square continuous.
</dir>



<dir class="definition"><b>(Mean square differentiable)</b>
A stochastic process $X_t$ is said to be mean square differentiable at $t$ if the following limit exists
$$ \lim\limits_{s \rightarrow t} \mathbb{E} \left[ \frac{|X_s-X_t|^2}{|s-t|^2}\right] < \infty$$
If $X_t$ is mean square differentiable at each $t\in \mathbb{T}$, then $X_t$ is said to be mean square differentiable and there exists a stochastic process $X_t'$ such that
$$ \lim\limits_{s \rightarrow t} \mathbb{E} \left[ (X_t' - \frac{X_s-X_t}{s-t})^2\right] = 0$$
</dir>

Also, other notions of differentiability can be defined, e.g., differentiability in probability, almost sure differentiability, differentiability in distribution, etc.

Let's consider two examples of stochastic processes, the Wiener and Cauchy processes.

##### Example: Wiener process

The Wiener process is mean square continuous which can be shown as follows:

By definition, we have that for $$X_s - X_t \sim \mathcal{N}(0, |s-t|)$$, we have that

$$\mathbb{E}[|X_s - X_t|^2] = \mathbb{E}_{X\sim \mathcal{N}(0, |s-t|)}[X^2] = |s-t|$$

Thus, the Wiener process is mean square continuous as
$$ \lim\limits_{s \rightarrow t} \mathbb{E}[|X_s-X_t|^2] = \lim\limits_{s \rightarrow t} |s-t| = 0$$

Let's now consider the mean square differentiability of the Wiener process. We have that

$$\mathbb{E} \left[ \frac{|X_s-X_t|^2}{|s-t|^2}\right] = \frac{|s-t|}{|s-t|^2} = \frac{1}{|s-t|}$$

As $s \rightarrow t$, this limit goes to infinity. Hence the Wiener process is not mean square differentiable.

##### Example: Cauchy process

The Cauchy process is not mean square continuous. We have that

$$\mathbb{E}[|X_s - X_t|^2] = \mathbb{E}_{X\sim \mathcal{C}(0, |s-t|)}[X^2] = \infty$$

simply because the Cauchy distribution has infinite variance. Hence the Cauchy process is not mean square continuous. 
Also, the Cauchy process is not mean square differentiable. 


We can observe this behavior in the interactive plots above, the Cauchy process has a much more volatile sample path than the Wiener process.


Anyway as we can see even this weaker notion of continuity and differentiability is not satisfied by many stochastic processes.
There are many more properties that a stochastic process can have, e.g., ergodicity, mixing properties, etc. We will not go into detail here, but we hope that this blog post gave you a good overview of the fascinating world of stochastic processes.

## Conclusion

In this blog post, we have introduced the concept of stochastic processes. We have seen that a stochastic process is a collection of random variables indexed by an arbitrary set. We have explored various properties of stochastic processes, such as finite-dimensional distributions, mean and covariance functions, stationarity, and equilibrium distributions. We have also discussed continuity and differentiability, focusing on mean square continuity and differentiability. We have illustrated these properties using examples of discrete-time and continuous-time stochastic processes. We hope that this blog post has provided you with a comprehensive understanding of stochastic processes and their properties.

