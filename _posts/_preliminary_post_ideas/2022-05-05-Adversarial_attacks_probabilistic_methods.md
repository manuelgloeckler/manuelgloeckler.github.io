---
layout: article
title: Adversarial attacks on probabilistic models
key: A8
---

## Adversarial attack
The by far most common domain of the "adversarial attack" community is classification. Thereby $f$ is a classifiers prediction function which predicts some label $y$. We typically distinguish a **targeted** from a **untargeted** attack. In the first case we have a specific target $t$ in mind. An adversarial attack is then typically defined as the following optimization problem
$$ \delta^* = \argmin_\delta || \delta || \text{  s.t.  } f(x+\delta) = t$$
In the second case we typically define it as
$$ \delta^* = \argmin_\delta || \delta || \text{  s.t.  } f(x+\delta) \neq  f(x)$$
i.e. we want a prediction that is different that that of $x$. This of course only makes sense if the classifiers prediction function is given as $f(x) = \argmax _y p(y|x)$.

This definition does only work for hard classification, but can easily be modified to classification, regression or essentially any other domain by rephrasing this problem as
$$ \delta^* = \argmin_\delta d(f(x+ \delta), t) \text{  s.t.  } || \delta ||\leq \epsilon $$
or for an untargeted attack
$$ \delta^* = \argmax_\delta d(f(x+ \delta), f(x)) \text{  s.t.  } || \delta ||\leq \epsilon $$
where $d$ is some measure of "distance" between the outputs of the network.

Because constraint optimization can be challenging an relaxation of this problem is often considered by instead solving
$$  \delta^* = \argmin_\delta d(f(x+ \delta), t) + \lambda || \delta || $$
or for an untargeted attack
$$ \delta^* = \argmax_\delta d(f(x+ \delta), f(x)) - \lambda || \delta || $$
where $\lambda$ is some regularization constant. 

Further one can distinguish **white box** and **black box** attacks. In the former we have full access to the prediciton function and can thus approximatly solve these optimization problems using numerical optimization methods as gradient decent. For the later we only have access to the input and outputs, thus have to solve it by "brute force" or optimization techniques like genetic algorithms or Bayesian optimization.

## Adversarial attacks for probabilistic models

Almost any machine learning model can be interpreted probabilistically i.e. binary classification can be modeled by a Bernoulli model, multi-class by an Categorical and regression by a Gaussian model. The underlying probabilistic model thus is typically determined by the choice of loss function. Bayesian methods on the other hand explicitly are constructed probabilistically. Thus the output of an neural network typically is a random variable/distribution over an output domain. Consequently the distance $d$ to determine an adversarial attack can take two forms:
* **An statistical divergence/distance measure**: For example the KL-divergence 
 $$ d(x, x+\delta) = D_{KL}(f(x)||f(x+\delta))$$
* **An distance measure based on a point estimate**: Be $f(x)=p(y|x)$. Then one typicall point estimate would be $g(x) = \mathbb{E}_y[f(x)]$. And we can define 
  $$d(x, x+\delta) = || g(x) - g(x+\delta)||$$

Which $d$ you choose depends the goal of the "adversarial attack". If you want to attack the distributional approximation, then the first method should be prefered (the choice of divergence will determine in what way you will deform it ;) ) . On the other hand if you are just interested in modifing the "predictions" then the second method may be better (and often easier). 

In any case the definition of "adversarial" stayes the same. 

## A flaw within these definitions

As introduced these definitions mainly stem from the classification domain, especially **image classification**. There it is clear that adding a small pertubation to an image should not change the prediction, because the result is more or loss indistinguishable for humans. Yet this may not hold true in generall, because clearly we generally have $p(y|x) \neq p(y| x + \delta)$. 

This implies there exist an irreducible adversarial vulnerability i.e. $d_f(x, x+\delta) \geq \epsilon$ with $\epsilon = d_{f^*}(x, x+\delta)$ where $f^*(x) = p(y|x)$ i.e. is the ground truth. This may be relevant in adversarial defense methods, because if these methods keep $d_f(x, x+\delta) < \epsilon$ then they must increase the generalization error! So we get robustness at the cost of "accuracy" ... Unfortunatly this threshold is generally intractable.

Another problem is the norm constraint $||\delta||$. Standard p-norm constraint may not be realistic. Further they do not include sematic meanings, which may especially be relevant in the context of images. For example in MNIST their certainly exists a small pertubtaion, which transforms a 9 into an 8 by just drawing a another half-circle in the lower left corner. Would you still consider this an "adversarial attack"? In the end $x + \delta$ now clearly is an 8 for us, so the classifier is right by predicting a different label, right?

This leads us to the controversial if there exists an tradeoff between accuracy and robustness. There are several optinions:
* There is a unavoidable tradeoff between accuracy an robustness [5]
* Robust and accurate models exist, we just do not know how to find them [4]

In generall the latter requires us to restirct us to so called **consistent pertubations** [4]. We call a pertubation $\delta$ consisitent if and only if
$$ p(\cdot |x) = p(\cdot| x + \delta)$$
It should be clear that such pertubations have zero irreducible adversarial vulnerability. Especially in image classification such pertubations are easy to find i.e. all rotations, horizontal flip, brightness or contrast changes or even small $\ell_p$ pertubations. In natural language word synonmy replacements come in mind, but in generall it is hard to get consistent pertubations for general models. 

We can relax this by just restricting $D(p(y|x)|| p(y|x+\delta)) \leq \epsilon$ and thus can define **consistent adversarial attacks** as 

$$ \delta^* = \argmin_\delta d(f(x+ \delta), t) \text{  s.t.  } D(p(y|x)|| p(y|x+\delta)) \leq \epsilon $$
or for an untargeted attack
$$ \delta^* = \argmax_\delta d(f(x+ \delta), f(x)) \text{  s.t.  } D(p(y|x)|| p(y|x+\delta))\leq \epsilon $$

Unfortunatly this is intractable, because we generally do not know $p(y|x)$ at all...

Alternatively, notice that if $f$ is optimal then $D(f(x + \delta)|| p(y|x+\delta)) = 0$. Thus we can define an adversarial attack as a pertubation where $f$ is not optimal, but the predicition are very different. 

$$ \delta^* = \argmin_\delta  d(f(x+\delta, t), f(x)) + D(f(x + \delta)|| p(y|x+\delta))$$

These adversarial attack thus tries to avoid using the irreducible adversarial robustness of a given model, but only the "bugs" within our model!

## Example: Gaussian conditional density

Consider the conidtional density 
$$ p(y|x) = \mathcal{N}(y; x, \sigma^2 I) $$
The natural adversarial robustness of this model according to the KL divergence is given by
$$ r_{nat}(\delta|x) = D_{KL}(p(y|x)|| p(y|x + \delta)) = \frac{1}{2\sigma^2} \delta^T \delta + const = \frac{1}{2\sigma^2} || \delta||_2^2$$
Thus we can get the set of consistent pertubations only if
$$ r_{nat}(\delta|x) \overset{!}{=} 0 \iff ||\delta||_2^2 = 0 \text{ or } \sigma^2 \rightarrow \infty$$
Thus the set of consistent pertubations equals $C(\delta) = \{ 0 \}$ for any finite variance or $C(\delta) = \mathbb{R}^d$ in the limit of $\sigma^2 \rightarrow \infty$. As we can see this criterium is in such case much to strict!

So let's relax this assumpiton as proposed. We obtain
$$ r_{nat}(\delta|x) \leq \epsilon \iff || \delta||_2^2 \leq 2 \sigma^2 \epsilon$$
Notice that this is just a norm constraint. So we can define the set of **$\epsilon$-consistent pertubations** as
$$ C_\epsilon(\delta) = \{ \delta \in \mathbb{R}^d : || \delta||_2^2 \leq 2\sigma^2 \epsilon \}$$

On the other hand if we slightly modify the problem to
$$ p(y|x) = \mathcal{N}(y; x, \Sigma) $$
this changes. Especially we have that
$$ r_{nat}(\delta|x) \leq \epsilon \iff \delta^T \Sigma^{-1}\delta \leq  2\epsilon \iff ||\delta||_{\Sigma^{-1}}^T \leq 2 \epsilon$$
which is the Mahalonobis distance. Thus the set of $\epsilon$ consistent pertubations equals the $2\epsilon$ ball induced by the Mahalonobis distance.

Notice that this problem is extremly simple. Because if we want to estimat $y = f(x)$ the optimal function is just the identity...

## Example: Gaussian conditional density II (Regression)
We now consider
$$ p(y|x) = \mathcal{N}(y; f(x), \sigma^2 I) $$
for some function $f$. The natural adversarial robustness is given by
$$ r_{nat}(\delta|x) = D_{KL}(p(y|x)|| p(y|x + \delta)) = \frac{1}{2\sigma^2} (f(x) - f(x+\delta))^T (f(x) - f(x+\delta)) + const = \frac{1}{2\sigma^2} ||f(x) - f(x+\delta)||_2^2= \frac{1}{2\sigma^2} (||f(x)||_2^2 + || f(x + \delta)||_2^2 - 2 f(x)^T f(x+\delta))$$
Again consistent observation may be to strict, the set of them is given by
$$ C(\delta|x) = \{ \delta | f(x) = f(x + \delta) \}$$
In general this set is very hard to get. Even obtaining single elements will be hard. However we can get an subset of it, which more or less contains all $\delta$ for which this holds around $x$. Consider a Taylor expansion of $f$, then $f(x + \delta) \approx f(x) + J_f(x) \delta$. Thus the equation reduce to the following
$$ \tilde{C}(\delta|x) = \{ \delta  | J_f(x)\delta = 0 \} = null(J_f(x))$$
Hence we can atleast get an local approximaiton of consistent pertubations, which is simply given by the null space of the Jacobian. We can proof that this is indeed a subset $\tilde{C}(\delta|x) \subset C(\delta|x)$ by proofing that any higher order term of the Taylor expansion are zero for any $\delta$ within the null space. ( not sure if this is true, but it kinda makes sense).

NOTE: WE CAN DO HERE THE FISHER APPROXIMATION TOO!!! By an empirical non parameteric estimate of the simulators fisher information matrix

We now again can get the $\epsilon$-consistent pertubatuions, which can be derived equivalently.

**PROBLEM: THIS REQUIRES FULL KNOWLEDGE ABOUT THE GROUND TRUTH...**


## References

* [1] Adversarial Attacks on Probabilistic Autoregressive Forecasting Models
* [2] Adversarial Robustness of Flow-Based Generative Models
* [3] Adversarial Attacks on Variational Autoencoders
* [4] Understanding and Mitigating the Tradeoff Between Robustness and Accuracy
* [5] Theoretically Principled Trade-off between Robustness and Accuracy