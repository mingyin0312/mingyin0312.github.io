---
layout: distill
title: Flow models for Generative AI 
description: As an alternative to Diffusion Models, Continuous Normalizing Flow Matching is one of the most powerful paradigm for generative AI modeling.  
date: 2024-08-20 

authors:
  - name: Ming Yin, Mengdi Wang
    url: 
    affiliations:
      name: Princeton ECE


bibliography: flow-distill.bib


---

A **flow** formalizes the idea of the motion of particles in a fluid and it is fundamental to the study of ordinary differential equations (ODEs). A flow may be viewed as a continuous motion of points over time, and they are ubiquitous in science, including engineering and physics. In modern AI era, flow models find its own shining point since NeuralODE <d-cite key="chen2018neural"></d-cite> that describes a family of deep learning models with nice properties. Recently, generative AI has elevated the power of AI to new levels. The development of flow models makes them suitable for generation, enhancing their relevance in generative AI. In this post, we discuss how flow models work from the methodology perspective.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/sd3_elevator.png">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/sd3_fox.png">
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/sd3_river.png">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/sd3_ufo.png">
    </div>
</div>
<div class="caption">
    My generations using Stable Diffusion 3 <d-cite key="esser2024scaling"></d-cite> with Rectified Flow models <d-cite key="liu2022flow"></d-cite> being its building blocks.
</div>


## Preliminaries

### ODE and Continuity Equations 


Let the data point $x=(x^1,\ldots,x^d)\in\mathbb{R}^d$. A *probability density path* $p:[0,1]\times \mathbb{R}^d\rightarrow \mathbb{R}_{>0}$ satisfies $\int p_t(x)dx=1$. A time-dependent vector field $v:[0,1]\times \mathbb{R}^d\rightarrow \mathbb{R}^d$ defines a (time-dependent diffeomorphic) flow with $\phi:[0,1]\times \mathbb{R}^d\rightarrow\mathbb{R}^d$ via the ODE:


$$
\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x)),\quad \phi_0(x)=x.
$$

density $p_0$ evolves to $p_1$ with the push-forward operator 

$$
p_t=[\phi_t]_{*}p_0,\quad [\phi_t]_{*}p_0(x) = p_0(\phi_t^{-1}(x))\mathrm{det}\left[\frac{\partial \phi_t^{-1}}{\partial x}(x)\right].
$$ 

Then the following theorem holds.

<hr>

**Theorem** A vector field $v_t$ is said to generate a probability density path $p_t$ if its flow $\phi_t$ satisfies the continuity equation 

$$
\frac{d}{dt}p_t(x)+\mathrm{div}(p_t(x)v_t(x))=0.
$$

<hr>


**Proof [Adopted from <d-cite key="bworld"></d-cite>]** Since $p_t=[\phi_t]_{*}p_0$, by change of variables for any measurable fucntion $g$ we have $\int g(\phi_t(y))p_t(y)dy=\int g(y)p_0(x)dx$. Next, by ODE we have $\partial_t\phi_t = v_t\circ \phi_t$.

For any test function $\psi\in\mathcal{C}_c^\infty((0,1]\times \mathbb{R}^n)$, it suffices to show for any $T\in(0,1]$

\begin{equation}\label{eq:int_ce}
    \int_0^T\int_{\mathbb{R}^n}\psi_t\partial_tp_tdxdt = -\int_0^T\int_{\mathbb{R}^n}\psi_t\mathrm{div}(p_t\cdot v_t)dx dt
\end{equation}

By integration by parts, we have 

$$
\left[\int_{\mathbb{R}^n} \psi_t p_t d x\right]_{t=0}^{t=T}-\int_0^T \partial_t \psi_t p_t d x d t=\int_0^T \nabla \psi_t \cdot v_t p_t d x d t,
$$

hence \eqref{eq:int_ce} is equivalent to 

$$
\int_{\mathbb{R}^n} \psi_t p_T d x-\int_{\mathbb{R}^n} \psi_t p_0 d x=\int_0^T \int_{\mathbb{R}^n}\left(\partial_t \psi_t\right) p_t+v_t \cdot \nabla \psi_t p_t d x d t.
$$

To show this, compute the derivate to obtain 

$$
\frac{d}{d t} \int_{\mathbb{R}^n} \psi_t p_t d x  =\frac{d}{d t} \int_{\mathbb{R}^n} \psi_t\left(t, \phi_t(y)\right) p_0(y) d y=\int_{\mathbb{R}^n} \frac{d}{d t} \psi\left(t, \phi_t(y)\right) p_0(y) d y 
$$


$$
\text { (chain rule) }  =\int_{\mathbb{R}^n}\left[\partial_t \psi_t\left(t, \phi_t(y)\right)+\nabla \psi_t\left(t, \phi_t(y)\right) \cdot \partial_t \phi_t(y)\right] p_0(y) d y 
$$

$$
\left(\partial_t \phi_t=v_t \circ \phi_t\right) =\int_{\mathbb{R}^n}\left[\partial_t \psi_t\left(t, \phi_t(y)\right)+\nabla \psi_t\left(t, \phi_t(y) \cdot v_t\left(\phi_t(y)\right)\right] p_0(y) d y\right. 
$$

$$
\text { (change of variables) } =\int_{\mathbb{R}^n}\left[\partial_t \psi_t(t, x)+\nabla \psi_t(t, x) \cdot v_t(x)\right] p_t(x) dx
$$

where the last line assgins $g(x)=\partial_t \psi_t(t, x)+\nabla \psi_t(t, x) \cdot v_t(x)$. Integrate both sides from $0$ to $T$ to complete the proof.


### Probability Flow ODE 

**Theorem (PF ODE)** For stochastic differential equation (SDE) with the form (where $\mathbf{f}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^d$, $\mathbf{G}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^{d \times d}$ and $\mathbf{w}$ be the $d$-dimensional Brownian motion):

\begin{equation}\label{eqn:SDE}
\mathrm{d} \mathbf{x}=\mathbf{f}(\mathbf{x}, t) \mathrm{d} t+\mathbf{G}(\mathbf{x}, t) \mathrm{d} \mathbf{w},
\end{equation}

and let its marginal probability density be $p_t(\mathbf{x}(t))$, then the following probability flow ODE 

$$
\mathrm{d} \mathbf{x}=\tilde{\mathbf{f}}(\mathbf{x}, t) \mathrm{d} t
$$ 

is also distributed according to $p_t(\mathbf{x}(t))$, given the same initial condition. Here $\tilde{\mathbf{f}}(\mathbf{x}, t):=\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\frac{1}{2} \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$.

**Remark: [special case]** For the simplified process 

$$
\mathrm{d} \mathbf{x}=\mathbf{f}(\mathbf{x}, t) \mathrm{d} t+g(t) \mathrm{d} \mathbf{w}
$$

where $g(\cdot): \mathbb{R} \rightarrow \mathbb{R}$ is a scalar function, its probability flow ODE 

$$
\mathrm{d} \mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} \mathrm{d} t.
$$


**Proof [Adopted from <d-cite key="song2020score"></d-cite>]** Since the SDE's \eqref{eqn:SDE} marginal probability density $p_t(\mathbf{x}(t))$ evolves according to Kolmogorov’s forward equation <d-cite key="KFE"></d-cite>

$$
\frac{\partial p_t(\mathbf{x})}{\partial t}=-\sum_{i=1}^d \frac{\partial}{\partial x_i}\left[f_i(\mathbf{x}, t) p_t(\mathbf{x})\right]+\frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \frac{\partial^2}{\partial x_i \partial x_j}\left[\sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t) p_t(\mathbf{x})\right],
$$

hence the above can be rewritten as 

$$
\begin{aligned}
\frac{\partial p_t(\mathbf{x})}{\partial t} & =-\sum_{i=1}^d \frac{\partial}{\partial x_i}\left[f_i(\mathbf{x}, t) p_t(\mathbf{x})\right]+\frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \frac{\partial^2}{\partial x_i \partial x_j}\left[\sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t) p_t(\mathbf{x})\right] \\
& =-\sum_{i=1}^d \frac{\partial}{\partial x_i}\left[f_i(\mathbf{x}, t) p_t(\mathbf{x})\right]+\frac{1}{2} \sum_{i=1}^d \frac{\partial}{\partial x_i}\left[\sum_{j=1}^d \frac{\partial}{\partial x_j}\left[\sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t) p_t(\mathbf{x})\right]\right]
\end{aligned}
$$

Since

$$
\begin{aligned}
& \sum_{j=1}^d \frac{\partial}{\partial x_j}\left[\sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t) p_t(\mathbf{x})\right] \\
= & \sum_{j=1}^d \frac{\partial}{\partial x_j}\left[\sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t)\right] p_t(\mathbf{x})+\sum_{j=1}^d \sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t) p_t(\mathbf{x}) \frac{\partial}{\partial x_j} \log p_t(\mathbf{x}) \\
= & p_t(\mathbf{x}) \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]+p_t(\mathbf{x}) \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x}),
\end{aligned}
$$

denote $\tilde{\mathbf{f}}(\mathbf{x}, t):=\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\frac{1}{2} \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, then 

$$
\begin{aligned}
\frac{\partial p_t(\mathbf{x})}{\partial t}= & -\sum_{i=1}^d \frac{\partial}{\partial x_i}\left[f_i(\mathbf{x}, t) p_t(\mathbf{x})\right]+\frac{1}{2} \sum_{i=1}^d \frac{\partial}{\partial x_i}\left[\sum_{j=1}^d \frac{\partial}{\partial x_j}\left[\sum_{k=1}^d G_{i k}(\mathbf{x}, t) G_{j k}(\mathbf{x}, t) p_t(\mathbf{x})\right]\right] \\
= & -\sum_{i=1}^d \frac{\partial}{\partial x_i}\left[f_i(\mathbf{x}, t) p_t(\mathbf{x})\right] \\
& +\frac{1}{2} \sum_{i=1}^d \frac{\partial}{\partial x_i}\left[p_t(\mathbf{x}) \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]+p_t(\mathbf{x}) \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] \\
= & -\sum_{i=1}^d \frac{\partial}{\partial x_i}\left\{f_i(\mathbf{x}, t) p_t(\mathbf{x})\right. \\
& \left.-\frac{1}{2}\left[\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]+\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] p_t(\mathbf{x})\right\} \\
= & -\sum_{i=1}^d \frac{\partial}{\partial x_i}\left[\tilde{f}_i(\mathbf{x}, t) p_t(\mathbf{x})\right]
\end{aligned}
$$

which is the Kolmogorov’s forward equation for the ODE

$$
\mathrm{d} \mathbf{x}=\tilde{\mathbf{f}}(\mathbf{x}, t) \mathrm{d} t.
$$ 


## Flow Matching <d-cite key="lipman2022flow"></d-cite>

The beginning setup:

<ul>
    <li> Data $x_1$ distributed according to some unknown data distribution $q(x_1)$;
   </li>
    <li>
    Probability path $p_t$ satisfies $p_0=p=\mathcal{N}(x|0,I)$;
    </li>
    <li>
      $p_1$ is the roughly equal approximation of the data distribution $q$.
  </li>
</ul>
The Flow Matching objective is then designed to match this target probability path, which will allow us to flow from $p_0$ to $p_1$.


### FM Objective 

Let $u_t(x)$ be the corresponding vector of the probability density path $p_t(x)$, then Flow Matching objective is defined as 

$$
\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t, p_t(x)}\left\|v_t(x)-u_t(x)\right\|^2
$$

where $\theta$ is the parameter of $v_t(x,\theta)$, $t\sim \mathrm{Unif}[0,1]$ and $x\sim p_t(x)$. In general, the above flow matching loss is intractable as $u_t,p_t$ are unknowns. However, for a data sample $x_1$, we can model the conditional probabiltiy distribution $p_t(x\|x_1)$ to satisfy $p_0(x\| x_1)=p(x)$ and $p_0(x\|x_1)=\mathcal{N}(x\|x_1,\sigma^2I)$ with $\sigma^2$ sufficiently small. Then we can recover the marginal path 

$$
p_t(x)=\int p_t\left(x \mid x_1\right) q\left(x_1\right) d x_1,
$$

with $p_1(x)=\int p_1\left(x \mid x_1\right) q\left(x_1\right) d x_1 \approx q(x)$. Importantly, let $u_t(x)$ be the vector field that generates $p_t(x)$ and $u_t(x\|x_1)$ be the vector field that generates $p_t(x\|x_1)$, then it follows 

\begin{equation}\label{eqn:equal} 
u_t(x)=\int u_t\left(x \mid x_1\right) \frac{p_t\left(x \mid x_1\right) q\left(x_1\right)}{p_t(x)} d x_1
\end{equation} 

**Proof of \eqref{eqn:equal}.** Notice that 

$$
\begin{aligned}
\frac{d}{d t} p_t(x) & =\int\left(\frac{d}{d t} p_t\left(x \mid x_1\right)\right) q\left(x_1\right) d x_1=-\int \operatorname{div}\left(u_t\left(x \mid x_1\right) p_t\left(x \mid x_1\right)\right) q\left(x_1\right) d x_1 \\
& =-\operatorname{div}\left(\int u_t\left(x \mid x_1\right) p_t\left(x \mid x_1\right) q\left(x_1\right) d x_1\right)=-\operatorname{div}\left(u_t(x) p_t(x)\right).
\end{aligned}
$$
From Preliminaries section, we finish the proof. 

Given CF, we can consider the *Conditional Flow Matching* defined as follows

$$
\mathcal{L}_{\text {CFM }}(\theta)=\mathbb{E}_{t, q\left(x_1\right), p_t\left(x \mid x_1\right)}\left\|v_t(x)-u_t\left(x \mid x_1\right)\right\|^2.
$$

Furthermore, CFM loss not only make the objective tractable, more importantly, 
it is equivalent to optimize the FM loss, i.e. 
$$\nabla_\theta \mathcal{L}_{F M}(\theta)=\nabla_\theta \mathcal{L}_{C F M}(\theta)$$ (Theorem 2 of <d-cite key="lipman2022flow"></d-cite>). As a result, we work with $\mathcal{L}_{\text {CFM }}(\theta)$ for the rest of the article.  



### Guassian Conditional Probability Paths

We model the conditional probability paths via Guassian. Concretely, we consider

$$
p_t\left(x \mid x_1\right)=\mathcal{N}\left(x \mid \mu_t\left(x_1\right), \sigma_t\left(x_1\right)^2 I\right)
$$ 

with $\mu:[0,1]\times \mathbb{R}^d\rightarrow \mathbb{R}^d$ and $\sigma:[0,1]\times \mathbb{R}\rightarrow \mathbb{R}$ to be the time-dependent mean and std. We set $\mu_0(x_1)=0,\sigma_0(x_1)=1$ and $\mu_1(x_1)=x_1,\sigma_1(x_1)=\sigma_{\mathrm{min}}$ small. The flow (there are infinite many of them we simply choose one) 

$$
\psi_t(x\mid x_1)=\sigma_t\left(x_1\right) x+\mu_t\left(x_1\right)
$$

generates $$
p_t\left(x \mid x_1\right)=\mathcal{N}(x \mid \mu_t\left(x_1), \sigma_t\left(x_1\right)^2 I\right)
$$  since $$\left[\psi_t\right]_* p(x)=p_t\left(x \mid x_1\right)$$. Notice the corresponding conditional vector field $u_t(x|x_1)$ satisfies 

$$
\frac{d}{d t} \psi_t(x)=u_t\left(\psi_t(x) \mid x_1\right),
$$

we can explicitly solve that (Theorem 3 of <d-cite key="lipman2022flow"></d-cite>)

$$
u_t\left(x \mid x_1\right)=\frac{\sigma_t^{\prime}\left(x_1\right)}{\sigma_t\left(x_1\right)}\left(x-\mu_t\left(x_1\right)\right)+\mu_t^{\prime}\left(x_1\right).
$$

By Reparameterizing $p_t(x\mid x_1)$ in terms of just $x_0$ in the CFM loss we get

$$\label{eqn:CFM_obj}
\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q\left(x_1\right), p\left(x_0\right)}\left\|v_t\left(\psi_t\left(x_0\right)\right)-\frac{\sigma_t^{\prime}\left(x_1\right)}{\sigma_t\left(x_1\right)}\left(x_0-\mu_t\left(x_1\right)\right)-\mu_t^{\prime}\left(x_1\right)\right\|^2.
$$

### Training and Sampling 


<ul>
    <li> Training: replacing $x_1,x_0$ in $\mathcal{L}_{\mathrm{CFM}}(\theta)$ with samples $x_1\sim q(x_1),x_0\sim p(x_0)$ and train the empirical objective;
   </li>
    <li>
    Sampling: first draw a noise sample $x_0\sim p(x_0)=\mathcal{N}(0,I)$, then compute $\phi_1(x_0)$ via solving $$
\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x)),\quad \phi_0(x)=x.
$$

via ODE solvers.
    </li>
</ul>


### Examples

**Diffusion conditional vector field.** Variance Exploding path: $p_t(x\mid x_1)=\mathcal{N}\left(x \mid x_1, \sigma_{1-t}^2 I\right)$, and the vector field $u_t\left(x \mid x_1\right)=-\frac{\sigma_{1-t}^{\prime}}{\sigma_{1-t}}\left(x-x_1\right)$. 

**Optimal Transport conditional vector field.** $\mu_t(x)=t x_1$, $\sigma_t(x)=1-\left(1-\sigma_{\min }\right) t$, and the vector field $u_t\left(x \mid x_1\right)=\frac{x_1-\left(1-\sigma_{\min }\right) x}{1-\left(1-\sigma_{\min }\right) t}$. The CFM loss takes the form 

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q\left(x_1\right), p\left(x_0\right)}\left\|v_t\left(\psi_t\left(x_0\right)\right)-\left(x_1-\left(1-\sigma_{\min }\right) x_0\right)\right\|^2
$$

which is very close to the Rectified flow models in the following section.



## Rectified Flow <d-cite key="liu2022flow"></d-cite>

**Methods** Given empirical observations of $X_0\sim\pi_0,X_1\sim \pi_1$, the rectified flow induced from $(X_0,X_1)$ is an ODE $dZ_t=v(Z_t,t)dt$ that converts $Z_0\sim \pi_0$ to $Z_1\sim \pi_1$. The vector field $v: [0,1]\times\mathbb{R}^d \rightarrow \mathbb{R}^d$ is set to drive the flow to follow the direction $X_1-X_0$ via a linear path:

$$
\min_v \int_0^1 \mathbb{E}\left[\left\|\left(X_1-X_0\right)-v\left(X_t, t\right)\right\|^2\right] \mathrm{d} t, \quad \text { with } \quad X_t=t X_1+(1-t) X_0,
$$

Clearly, $dX_t=(X_1-X_0)dt$, which makes the process $X_t$ non-causal. 


**Training and Sampling** With empirical draws of $(X_0,X_1)$, we solve the above objective and get $v$. After getting $v$, we solve the ODE either starting from $Z_0\sim \pi_0$ to transfer $\pi_0$ to $\pi_1$, or backwardly starting from $Z_1\sim \pi_1$ to transfer $\pi_1$ to $\pi_0$. The obtained flow $(Z_0^k,Z_1^{k})$ can be used as input to reflow and obtain $(Z_0^{k+1},Z_1^{k+1})$ (see Algorithm 1 of <d-cite key="liu2022flow"></d-cite> for details).


**A Nonlinear Extension** Let $X=\\{X_t: t \in[0,1] \\}$ be any time-differentiable random process that connects $X_0$ and $X_1$. Let $\dot{X}_t$ be the time derivative of $X_t$. The (nonlinear) rectified flow induced from $X$ is defined as ($w_t$ is a positive weight sequence) 

$$
\mathrm{d} Z_t=v^{\boldsymbol{X}}\left(Z_t, t\right) \mathrm{d} t, \quad \text { with } \quad Z_0=X_0, \quad \text { and } \quad v^{\boldsymbol{X}}(z, t)=\mathbb{E}\left[\dot{X}_t \mid X_t=t\right].
$$

[Theorem 3.3, 3.5-7 of <d-cite key="liu2022flow"></d-cite>] $X$ is rectifiable if $v^X$ is locally bounded and the solution of the integral equation below exists and is unique: 

$$Z_t=Z_0+\int_0^t v^{\boldsymbol{X}}\left(Z_t, t\right) \mathrm{d} t, \quad \forall t \in[0,1], \quad Z_0=X_0$$

<ul>
    <li>Assume $X$ is rectifiable and $Z$ is its rectified flow. Then $$\operatorname{Law}\left(Z_t\right)=\operatorname{Law}\left(X_t\right) \text { for } \forall t \in[0,1]$$
   </li>
    <li>
    Assume $(X_0, X_1)$ is rectifiable and $\left(Z_0, Z_1\right)=\texttt{Rectify}\left(\left(X_0, X_1\right)\right)$, then for any convex function $c: \mathbb{R}^d \rightarrow \mathbb{R}$, then 

    $$
    \mathbb{E}\left[c\left(Z_1-Z_0\right)\right] \leq \mathbb{E}\left[c\left(X_1-X_0\right)\right]
    $$

    </li>
    <li>
        Let $Z^k$ be the k-th rectified flow induced from $(X_0, X_1)$. Let the straightness be $S({Z})=\int_0^1 \mathbb{E}\left[\left\|\left(Z_1-Z_0\right)-\dot{Z}_t\right\|^2\right] d t$. Then

        $$
        \min _{k \in\{0 \cdots K\}} S\left(\boldsymbol{Z}^k\right) \leq \frac{\mathbb{E}\left[\left\|X_1-X_0\right\|^2\right]}{K}
        $$

    </li>
</ul>

## Other Flow Model Recipes

There are other different flow matching models that either improve the performance or the efficiency such as consistency model matching <d-cite key="yang2024consistency"></d-cite>, conditional flow matching <d-cite key="tong2023improving"></d-cite>, and latent flow matching <d-cite key="dao2023flow"></d-cite>.












<!-- 

****

****
 -->

<!-- ## Miscellaneous

My nice collaborator also shared this on twitter: 
{% twitter https://twitter.com/yubai01/status/1358887058274570241 maxwidth=500 max=5 %} -->