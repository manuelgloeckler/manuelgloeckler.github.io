---
layout: article
title: Information geometry
key: A5
tags: Math
cover: https://www.researchgate.net/publication/351854812/figure/fig1/AS:1132455809564700@1647009913987/Representation-of-a-2D-statistical-manifold-with-its-tangent-space-at-a-point-th-and-two_Q640.jpg
comment: true
---

Information geometry can be thought as part of the *Information sciences* which study the "communication" between data and families of models. In short, they seek to distill information from data to models. This includes information theory but also the fields of probability theory, statistics, machine learning and more. In contrast to these fields however they investigate this geometrically, especially using concepts from differential geometry and riemannian manifolds.


## A parametric statistical model as Manifold

A parameteric family of probability distributions is a set of distributions $\mathcal{F}= \{ p_\theta (x) \mid \theta \in \Theta \}$ labeld by the parameters $\theta$. Typically $\theta \in \mathbb{R}^n$ (or at least we can typically reparamterize them accordingly) and thus each family $\mathcal{F}$ forms a *statistical manifold* in $\mathbb{R}^n$, namely a space in which each point,labeld by coordinates $\theta$, represents a probability distribution $p_\theta (x)$. By interpreting $\mathcal{F}$ as manifold $\mathcal{M}$ one hopes to get some insights into the structure of such a family. For example, one might hope to discover a reasonable measure of "distance" of two distributions within the family.

But wait, what we actually mean by calling it a "manifold". This is a very overused term and typically means informally: "Some lower dimensional subspace" i.e. natural pictures in the space of all pictures. This definition, maybe nice for intuition but doesn't realy help us here. We need something more mathematical... . So let's review the basics.

Warning: I did not study topology! This will not be very formal ... (but may also is prefered by some ;) )


### Manifolds

We live in an Euclidean space and thus the average human intuition is based on a euclidean geometry. For me (and most other people) a cup is different from a donut. If we rotate a donut it stays the same, if we rotate a cup it becomes different... . For a (very strict) topologist however this objects are the same. The reason for this is because the exist a smooth bijection which maps any point within a donut, to a point within a cup. This is demonstrated below

<figure>
  <p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Mug_and_Torus_morph.gif" />
  </p>
  <figcaption> Homeomorphism between "cup" and "donut" i.e. an smooth invertible mapping between both geometrical objects. </figcaption>
</figure>



A topologist call these geometric forms **homeomorphic** i.e. topologically equivalent. Every property which holds for a donut, also holds for a cup.

An **euclidean space** is typically a vector space equipped with an inner product. Thus we know exactly how to measure distances between point or angles between lines.

An **topological space** on the other hand is defined much more general. Roughly speaking only the closeness is defined but cannot neccessarily be measured by a numerical distance i.e. it is just a set of point where each point is associated with a neighborhood, satisfying some axioms which essentially just relate points to their neighborhoods. As an euclidean space is equiped with a measure of distance it trivially is a topological space. The neighborhood around $x$ e.g. is just the collection of points which are a certain distance aways from it.

A **manifold** tries to combine the best of both worlds, the generality of topological spaces and the nice properties of Euclidean space. We can define a manifold as a topological space which locally resembles an Euclidean space. Let's consider a line of a certain length, this is a geometrical object in one dimensional euclidean space $\mathbb{R}$. One way to embed it into $\mathbb{R}^2$ is to just stick the ends together leading to a circle (or some circle like structure). Note that these objects are not topologically equivalent, one can easy see this because if you remove one point on a line then it becomes disconnected  whereas a circle stays connected. Yet the circle and the line are locally homeomorphic i.e. at any point the transformation is homeomorphic around a sufficiently small neighborhood. The same holds for a 2d plane and a sphere. This is why for us the earth looks like a plane, we aren't large enough to see outside some neighborhood on which the earth actually does look like a plane. This is what is deomonstrated in the image below. Each arc of the circle is locally similar to a line segment and if we take the inifnitesimal limit it will locally resemble a line segment.

In other words the surface of a circle or a sphere is a one/two dimensional manifold in $\mathbb{R}^2$ and $\mathbb{R}^3$.Let's make this a bit more formal.

## Manifolds

### Basic notation and definitions
<dir class="definition">
We call  $\mathcal{M}$ an $n$ dimensional manifold, if for any $x \in \mathcal{M}$ there exists an open neighborhood $U$ and a homeomorphism $\psi: U \rightarrow V$ which maps $U$ onto an open set $V \subset \mathbb{R}^n$.

<ul>
<li> We call $\varphi$ the <b>chart</b>. </li>
<li> A set of charts $\{ \varphi_i : U_i \rightarrow V_i\}$ is called the <b>atlas</b> of $\mathcal{M}$ if $\bigcup_i U_i = \mathcal{M}$. </li>
</ul>
</dir>

So in simple words, manifolds are all about mappings i.e. the charts which maps any local neighborhood on the manifold to a "flat" Euclidean space. Let's look at one of the simplet examples of an manifold: The circle, which is given by the set $\mathbb{S} = \{(x,y) \mid x^2 + y^2 = 1\}$


<figure>
  <p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/64/Circle_with_overlapping_manifold_charts.svg"/>
  <figcaption> Four mappings i.e. 'charts', which map each arc of the circle to the open interval. </figcaption>
  </p>
</figure>


As visualized in figure 2 any arc of a circle can be mapped to an open interval $(-1, 1)$, by mapping each point of the arc to either it's $x$ (for top and bottom) or $y$ (for left, right) coordinate. Porjections onto the first coordinate is a continous and invertible mapping and thus these mappings are charts i.e. 

$$ \varphi_{top}(x,y) = x \quad \varphi_{bottom}(x,y) = x \quad \varphi_{left}(x,y) = y \quad \varphi_{right} =y$$ 

Together these parts cover the whole circle and thus they form an atlas of the whole manifold i.e.

$$ \mathcal{A}(\mathbb{S}) = \{\varphi_{top}, \varphi_{bottom}, \varphi_{left}, \varphi_{right}\}$$

This is not the only atlas of a circle, it is not even the smallest atlas one can find (you can come up one which only uses two charts).

We can thus map each point $x$ to a so-called *coordinate* $\varphi_\cdot (x)$ by choosing the right chart. Yet note that the charts we just created have overlapping domains i.e. the top chart overlaps with the right and left one. Thus any point lying in the intersection $(x,y) \in U_{top} \cap U_{left}$ can be mapped to two different coordinates $\varphi_{top} (x,y) = x$ and $\varphi_{right}(x,y) = y$. Clearly both coordinates must be related as they describe the same point on the manifold. We can make the relationship clear by introducing the so called **transition map**, which maps the coordinate from one chart to another on an intersecting domain. We charts must be invertible we can easily define them as following:

$$ T_{top, right}: V_{top} \rightarrow V_{right} =  \varphi_{right} \circ \varphi^{-1}_{top}$$

For a circle we can easily compute it

$$ T_{top, right}(a) = \varphi_{right}(\varphi_{top}^{-1}(x)) = \varphi_{right}(a, \sqrt{1-a^2}) = \sqrt{1-a^2}$$

These transition maps are important because depending on their differentiability they define a less general but nice class of manifolds: An **differentiable manifold**. If these are also infinitly often differentiable, we call them **smooth manifold**. The motivaition is that now we can translate all the nice concepts of calculus to manifolds.

### Tangent spaces

The tangent space is the the generalization of tanget lines or tangent planes you probably now from calculus to an differentiable manifold.

Let's consider a simple differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}$. From your high school calculus you know that the tangent line at any point $x$ is given by $T_x f = \{f(x) + f'(x) t \mid t \in \mathbb{R}\}$. From your undergrad studies you may come in contact with it's multivariate generalization. Be $f = \mathbb{R}^n \rightarrow \mathbb{R}^m$ then for any point $x \in \mathbb{R}^n$ and direction $v \in \mathbb{R}^n$ we can get the tangent line of $f$ in direction $v$ by $T_x f_v = \{f(x) + f_v'(x)t \mid t \in \mathbb{R} \}$, where $f_v'$ denotes the directional derivative. From this we can obtain the tangent plane by considering the linear space spanned by $n$ linearly indepedendent directions i.e. $e_1, \dots, e_n$. We obtain

$$ T_x f = \left\{f(x) + \sum_{i=1}^n f_{e_i}'(x)t_i \mid t_1, \dots, t_n \in \mathbb{R} \right\} = \left\{f(x) + J_f t \mid t \in \mathbb{R}^n \right\}$$

This is actually very close to how we define a tangent space on a manifold $\mathcal{M}$. Yet, there is one problem on a manifold we currently don't have the notion of directional derivatives or linear independent vectors that can span this space. The solution is to look at **curves** on the manifold, as demonstrated in the figure below.

<figure>
    <p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Tangentialvektor.svg/1920px-Tangentialvektor.svg.png" alt="Forest" style="width:50%">
    </p>
  <figcaption> A tangent space $T_x\mathcal{M}$ of a manifold $\mathcal{M}$ at some point $x$, together with it's tagent vector $v$ and a smooth curve $\gamma(t)$ on $\mathcal{M}$. </figcaption>
</figure>

Especially we require *smooth parameteric curves* $\gamma(t): [a,b] \rightarrow \mathcal{M}$, that takes some parameter $t$ and gives us some point on the manifold. As it is smooth it basically defines a curve that runs along the manifold. Yet, to use basic rules form calculus we require to embed this curve into an euclidean space. Luckily, we can do this by using the chart's and consider the curve within the local coordinates i.e. 

$$ y: [a,b] \rightarrow \mathbb{R}^d \text{ with } y(t) = \varphi(\gamma(t)).$$

This function is just a "normal" function within in euclidean space. Thus to get a *tangent vector*, we just can differentiate it with respect to $t$. Let's assume that our curve goes through our point of interest $x$, then there is an $t_0$ such that $\gamma(t_0) = x$. We obtain

$$ v_\gamma := \frac{dy}{dt}(t_0)  = \left( \frac{dy_1}{dt}(t_0), \dots, \frac{dy_d}{dt}(t_0) \right)$$

So, we just generalized the notion of 'directional derivaitves' to manifolds. Instead of considering vectors pointing in a certain direction, we consider curves that go throught $x$. As a result we can define the $T_x\mathcal{M}$ as the space spanned by all $v_{\gamma}$ for all curves $\gamma$ that go through $x$.

Let's try to compute tangent vectors on our circle. For this we need a parametric curve, which we can get by using polar coordinates i.e.

$$\gamma: [a,b] \rightarrow \mathbb{S}: \ \gamma(t) = (\cos(t), \sin(t)) $$

We thus obtain that the coordinates of this curve vary by

$$ y(t) = \varphi_{right}(\gamma(t)) = \sin(t) \qquad \frac{dy}{dt}(t_0) = \cos(t_0) $$

where $\gamma(t_0) = x$, thus $t_0 = arc\cos\left(\frac{x_0}{\sqrt{x_0^2 + x_1^2}}\right)$. All what is left is to map back to the manifold for which we can use the inverse chart. We obtain

$$ v_\gamma = \frac{dy}{dt}(t_0) = \cos(t_0) = \frac{x_0}{\sqrt{x_0^2 + x_1^2}}$$

Thus we can conclude that the tangent space of the circle at $x$ is given by

$$ T_x \mathbb{S} = \left\{ x + \frac{x_0}{\sqrt{x_0^2 + x_1^2}} t \mid t \in \mathbb{R} \right\} $$

as the manifold is one dimensional, the tangent space is one dimensional as well and we don't need to consider the other curves!


### Riemannian manifolds


A Riemannian manifold is equipped with a metric tensor, which assigns an inner product to each pair of tangent vectors at each point on the manifold. This inner product defines a notion of length and angle, allowing us to measure distances and angles intrinsically on the manifold.

#### The Riemannian Metric

The key ingredient that defines a Riemannian manifold is the Riemannian metric. A Riemannian metric on a smooth manifold $M$ is a smoothly varying family of inner products on the tangent spaces of $M$. Mathematically, it is a symmetric, positive-definite tensor field, usually denoted as $g$. The metric tensor $g$ assigns an inner product $\langle v, w \rangle$ to each pair of tangent vectors $v$ and $w$ at each point on the manifold.

<dir class="definition">
A Riemannian Metric on a smooth manifold $\mathcal{M}$ is defined as follows:

For each point $x$ in the manifold $\mathcal{M}$, there exists an open neighborhood $U$ and a smooth function $g_x: U \rightarrow \mathbb{R}^{n\times n}$ such that:

<ul>
<li> The function $g_x$ assigns a symmetric positive-definite matrix to each point $p$ in the neighborhood $U$, i.e., $g_x(p)$ is a symmetric positive-definite matrix for all $p \in U$. </li>
<li> The collection of such functions $\{ g_x : U \rightarrow \mathbb{R}^{n\times n} \}$ is smoothly varying, meaning that for any two overlapping neighborhoods $U_1$ and $U_2$ in $\mathcal{M}$, the corresponding metric tensors $g_{x_1}$ and $g_{x_2}$ defined on these neighborhoods must smoothly transition between each other, ensuring compatibility. </li>
</ul>
</dir>

The metric tensor allows us to define the length of curves on the manifold. Given a curve $\gamma: [a, b] \rightarrow M$, the length of the curve between points $\gamma(a)$ and $\gamma(b)$ is given by the Riemannian length:

$$
L(\gamma) = \int_{a}^{b} \sqrt{\langle \dot{\gamma}(t), \dot{\gamma}(t) \rangle} dt
$$

where $\dot{\gamma}(t)$ is the tangent vector to the curve at each point.

#### Geodesics

Geodesics are curves on a Riemannian manifold that locally minimize distance. They are analogous to straight lines in Euclidean space. Formally, a geodesic is defined as a curve $\gamma(t)$ such that the tangent vector $\dot{\gamma}(t)$ is parallel transported along the curve. In other words, $\nabla_{\dot{\gamma}} \dot{\gamma} = 0$, where $\nabla$ denotes the Levi-Civita connection associated with the Riemannian metric.

#### Curvature

The curvature of a Riemannian manifold measures how much the geometry of the manifold deviates from that of Euclidean space. It is encoded in the Riemann curvature tensor, which quantifies how the infinitesimal area spanned by two vectors changes as one moves around a small loop on the manifold.

## Statistical Manifolds

A main concern of theory of inference is the problem of updating probabilities when new information arises. Typically we pick the "best fitting" model out of a family of distributions and this arises many questions:

* What if we had picked a "neighboring" distributions?
* How can we distinguish one distribution from anther?
* Can we quantify the difference?
* What is the shortest path from one distribution ot another?

This are inherently **geometic** questions and thus fall into the domain of **information geometry**. More specifically, in previous chapter we exactly learned how to answer this questions for Manifolds. Thus by interpreting a parametric family of distributions as a manifold we can use the tools of differential geometry to answer this questions. This is the main idea of information geometry.


A **statistical manifold** is a smooth manifold, denoted as $\mathcal{M}$, whose points represent probability distributions. Each point on the manifold corresponds to a particular probability distribution, such as a Gaussian distribution, a Poisson distribution, or any other type of distribution.


# TODO Add example and 

## References

* https://math.stackexchange.com/questions/708634/is-an-infinite-line-the-same-thing-as-an-infinite-circle
* https://en.wikipedia.org/wiki/Topological_space
* https://bjlkeng.github.io/posts/manifolds/
* https://www.robots.ox.ac.uk/~lsgs/posts/2019-09-27-info-geom.html#:~:text=In%20brief%2C%20information%20geometry%20is,and%20unexpected%20connections%20between%20them.
* http://www.kevinoconnor.co/wp-content/uploads/2018/03/ATasteOfInformationGeometry.pdf
* https://scholarship.claremont.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1095&context=hmc_theses




