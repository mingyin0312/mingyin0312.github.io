---
layout: post
title:  A Brief Introduction to Influence Funtion Technique
date:   2021-03-15 
description: Influence function technique is powerful in that it provides a way to calculate efficiency bound for the semiparameteric estimation problems. 




---

This post explains how to calculate efficiency bound through influence function technique.

A statistic $$\phi(P) \in\mathbb{R}$$, which is a function of the distribution of the vector $$X$$, $$P(X)$$. Examples include $$\mathbb{E}[X], \text{Var}(X)$$. How to define the differentiability of $$\phi$$ at $$P^0$$? Let's mimic the first order Taylor Expansion for $$\phi$$ in the functional form:

$$
\phi(P)\approx \phi(P^0)+D\phi(P-P^0)
$$

How to define it properly?

<hr>

### 1. Frechet derivative at $$P^0$$

Frechet derivative at $$P^0$$ is defined as:

$$
\lim _{P \rightarrow P^{0}} \frac{||\left(\phi(P)-\phi\left(P^{0}\right)\right)-D \phi\left(P-P^{0}\right)||_U}{\left\|P-P^{0}\right\|_V}=0
$$

where $$D\phi$$ is a continuous linear functional with respect to some norm $$\|P\|$$.

**Example: $$L^2$$ norm on the space of densities.**

\begin{equation}\label{eqn:l2norm}
||P||=\sqrt{\int\left(d P / d P^{0}\right)^{2} d P^{0}}
\end{equation}

To handle the functional operation, we need the following representation theorem.

<hr>

### 2. Riesz representation theorem

Let $$\mathscr{H}$$ be any Hilbert space with inner product $$\langle\cdot, \cdot\rangle$$ and the induced norm.Then for any continuous linear functional $$\psi: \mathscr{H} \rightarrow \mathbb{R}$$, there is an element $$
x \in \mathscr{H}
$$ such that $$\forall y \in \mathscr{H}$$,


$$
\psi(y)=\langle x, y\rangle
$$

<hr>

### 3. Influence function

Take  $$\mathscr{H}$$ to be the space of all measurable functions of $$X$$ which have mean zero and finite variance under $$P^0$$. The inner product is defined as $$
\langle y, z\rangle=E^{0}[y(X) \cdot z(X)]=\operatorname{Cov}(y, z)
$$.

Recall form \eqref{eqn:l2norm} the the norm of distribution is defined in the relative density sense, therefor there exists another linear functional $$D^\prime \phi$$ such that 

$$
\begin{aligned}
D\phi(P-P^0)&=D^\prime\phi(\frac{dP}{dP^0}-\frac{dP}{dP^0})=D^\prime\phi(\frac{dP}{dP^0}-1)=\mathbb{E}^0[IF(X)\cdot\frac{dP}{dP^0}(X)]\\
&=\int IF(X)\cdot\frac{dP}{dP^0}(X)dP^0(X)\\
&=\int IF(X)\cdot dP(X)=\mathbb{E}[IF(X)]\\
\end{aligned}
$$

Hence we have 

\begin{equation}\label{eqn:IF}
\phi(P) \approx \phi\left(P^{0}\right)+\mathbb{E}[I F(X)].
\end{equation}

<hr>

### 4. Intuitive derivation of influence function

Suppose we construct the plug-in estimator $$\hat{\phi}=\phi(P_n)$$, where $$P_n$$ is the empirical distribution

$$
P_n=\frac{1}{n}\sum_{i=1}^n\delta_{X_i},
$$

then use the **linear property** of the functional, we have 

$$
\begin{aligned}
\widehat{\phi} &\approx \phi\left(P^{0}\right)+D \phi\left(P_{n}-P^{0}\right)=\phi\left(P^{0}\right)+\frac{1}{n} \sum_{i=1}^n D \phi\left(\delta_{X_{i}}-P^{0}\right)\\
&=\phi(P^0)+E_n[D\phi(\delta_{X_i}-P^0)].
\end{aligned}
$$

where $$E_n$$ is the sample average. This suggests 

$$
IF(X)=D\phi(\delta_X-P^0).
$$

<hr>

### 5. How to actually calculate influence function?

Using directional derivatives: consider families of distributions indexed by a scalar $$\theta$$ with $$P(X;\theta)$$. Then from \eqref{eqn:IF} we directly have 

$$
\frac{\partial \phi}{\partial \theta}=\frac{\partial}{\partial\theta}\mathbb{E}[IF(X)]=\int IF(X)\frac{\partial}{\partial\theta}dP(X;\theta).
$$

and normalize it to so that $$IF$$ has zero mean.

**Example.** 

<ul>
    <li> 
        $$
\phi(P)=\operatorname{Var}(X)=\int X^{2} d P-\left(\int X d P\right)^{2}
$$
Then 
$$
\begin{aligned}
\frac{\partial \phi}{\partial \theta}&=\int X^{2} \frac{\partial}{\partial \theta} d P-2\left(\int X d P\right) \int X \frac{\partial}{\partial \theta} dP\\
&=\int\left(X^{2}-2 E[X] \cdot X\right) \frac{\partial}{\partial \theta} dP
\end{aligned}
$$
Normalize it to get 
$$
I F(X)=X^{2}-2 E[X] \cdot X-\operatorname{Var}(X)+E[X]^{2}!
$$
   </li>

</ul>

<hr>

### 6. Why should we care about influence function?

Most importantly, it implies the asymptotic efficiency bound for estimating $$\phi$$ is 

$$
\lim_{n\rightarrow\infty} n\cdot \operatorname{Var}(\hat{\phi})\geq \operatorname{Var}(IF(X)),
$$

which provides a powerful alternative when Fisher information style argument is not applicable!

For details see [Asymptotic statistics](https://books.google.com/books?hl=en&lr=&id=UEuQEM5RjWgC&oi=fnd&pg=PP19&dq=Asymptotic+statistics.+&ots=mpOFUCd_KB&sig=j0h2_Tt9cXqgycuXYKTLRFwRjNo#v=onepage&q=Asymptotic%20statistics.&f=false){:target="\_blank"} and [Semiparametric theory and missing data](https://books.google.com/books?hl=en&lr=&id=xqZFi2EMB40C&oi=fnd&pg=PR8&dq=Semiparametric+theory+and+missing+data&ots=NG7_acMaHA&sig=leMg2Of1nq4k-VUw2hTrmnv3eMA#v=onepage&q=Semiparametric%20theory%20and%20missing%20data&f=false){:target="\_blank"}.

<hr>

### 7. An application

In Offline Policy Evaluation, the asympototic linear Cramer-Rao lower bound is obtained using Information Function technique for the first time! See Theorem 4.5 of paper [Bootstrapping Statistical Inference for Off-Policy Evaluation](https://arxiv.org/pdf/2102.03607.pdf){:target="\_blank"} for more details. 

<hr>

The content of this post mainly comes from [this note](https://scholar.harvard.edu/files/kasy/files/ifhandout.pdf){:target="\_blank"} of Maximilian Kasy.











