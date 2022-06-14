---
layout: article
title: Information geometry and it's application in Machine Learning
key: A5
tags: Math
cover: https://www.researchgate.net/publication/351854812/figure/fig1/AS:1132455809564700@1647009913987/Representation-of-a-2D-statistical-manifold-with-its-tangent-space-at-a-point-th-and-two_Q640.jpg
comment: true
---

Information geometry can be thought as part of the *Information sciences* which study the "communication" between data and families of models. In short, they seek to distill information from data to models. These include information theory but also the fields of probability theory, statistics, machine learning and more. In contrast to these fields however they investigate this geometrically, especially using 


## A notion of distance 

A main concern of theory of inference is the problem of updating probabilities when new information arises. Typically we pick the "best fitting" model out of a family of distributions and this arises many questions:

* What if we had picked a "neighboring" distributions?
* How can we distinguish one distribution from anther?
* Can we quantify the difference?
* What is the shortest path from one distribution ot another?

This are inherently **geometic** questions and thus fall into the domain of **information geometry**. More specifiall we will discuss here the notion of "distance" between probability distributions as well as the induced statistical manifolds.


## A parametric statistical model as Manifold

A parameteric family of probability distributions is a set of distributions $\mathcal{F}= \{ p_\theta (x) | \theta \in \Theta \}$ labeld by the parameters $\theta$. Typically $\theta \in \mathbb{R}^n$ (or at least we can typically reparamterize them accordingly) and thus each family $\mathcal{F}$ forms a *statistical manifold* in $\mathbb{R}^n$, namely a space in which each point,labeld by coordinates $\theta$, represents a probability distribution $p_\theta (x)$. By interpreting $\mathcal{F}$ as manifold $\mathcal{M}$ one hopes to get some insights into the structure of such a family. For example, one might hope to discover a reasonable measure of "distance" of two distributions within the family.

But wait, what we actually mean by calling it a "manifold". This is a very overused term and typically means informally: "Some lower dimensional subspace" i.e. natural pictures in the space of all pictures. This definition, maybe nice for intuition but doesn't realy help us here. We need something more mathematical... . So let's review the basics.

Warning: I did not study topology! This will not be very formal ... (but may also is prefered by some ;) )


### Manifolds

We live in an Euclidean space and thus the average human intuition is based on a euclidean geometry. For me (and most other peaple) a cup is different from a donut. If we rotate a donut it stays the same, if we rotate a cup it becomes different... . For a (very strict) topologist however this objects are the same. The reason for this is because the exist a smooth bijection which maps any point within a donut, to a point within a cup. This is demonstrated below

![image](https://upload.wikimedia.org/wikipedia/commons/2/26/Mug_and_Torus_morph.gif)

A topologist call these geometric forms **homeomorphic** i.e. topologically equivalent. Every property which holds for a donut, also holds for a cup.

An **euclidean space** is typically a vector space equipped with an inner product. Thus we know exactly how to measure distances between point or angles between lines. 

An **topological space** on the other hand is defined much more general. Roughlz speaking only the closeness is defined but cannot neccessarily be measure by a numeric distance i.e. it is just a set of point where each point is associated with a neighborhood, satisfzing some axioms which essentiallz just relate points to their neighborhoods. As an euclidean space is equiped with a measure of distance it trivially is a topological space.

A **manifold** tries to combine the best of both worlds, the generality of topological spaces and the nice properties of euclidean space. We can define a manifold as a topological space which locally resembles an Euclidean space. Let's consider a line of a certain length, this of course is a geometrical object in one dimensional euclidean space $\mathbb{R}$. One way to embed it into $\mathbb{R}^2$ is to just stick the ends together leading to a circle (or some circle like structure). Note that these objects are not topologically equivalent, one can easy see this because if you remove one point on a line then it becomes disconnected  whereas a circle stays connected. Yet the circle and the line are locally homeomorphic i.e. at any point the transformation is homeomorphic around a sufficiently small neighborhood. The same holds for a 2d plance and a sphere. This is why for us the earth looks like a plane, we aren't large enough to see outside some neighborhood on which the earth actually does look like a plane. In other words the boundary of a circle or a sphere is a one/two dimensional manifold in $\mathbb{R}^2$ and $\mathbb{R}^3$.




References
* https://math.stackexchange.com/questions/708634/is-an-infinite-line-the-same-thing-as-an-infinite-circle
* https://en.wikipedia.org/wiki/Topological_space
* https://bjlkeng.github.io/posts/manifolds/
* https://www.robots.ox.ac.uk/~lsgs/posts/2019-09-27-info-geom.html#:~:text=In%20brief%2C%20information%20geometry%20is,and%20unexpected%20connections%20between%20them.









## References

