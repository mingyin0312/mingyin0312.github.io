---
layout: distill
title: Optimal offline RL with the unified model-based framework
description: A model-based framework + singleton absorbing MDP technique achieves the optimal rate for several challenging offline tasks.
date: 2021-06-24 

authors:
  - name: Ming Yin
    url: 
    affiliations:
      name: CS, UCSB


bibliography: optimal-uniform-ope.bib


---

The post explains the idea of paper <d-cite key="yin2021optimal"></d-cite>.

## Brief background on several Offline Learning tasks


Historical data $\mathcal{D}=\left\lbrace (s_t^{(i)},a_t^{(i)},r_t^{(i)})\right\rbrace_{i\in[n]}^{t\in[H]} $ was obtained by logging policy $\mu$ and we can only use $\mathcal{D}$ to estimate the value of target policy $\pi$, *i.e.* $v^\pi$. Suppose we only assume knowledge about $\pi$ and $r_t^{(i)} = r_t(s_t^{(i)},a_t^{(i)})$. The goal of offline learning task is to find an *$\epsilon$-optimal policy* $\pi_\text{out}$, such that  

$$
\left\lVert V_1^{\pi^\star}-V_1^{\pi_\text{out}}\right\rVert_\infty<\epsilon.
$$

Especially, <d-cite key="yin2021near"></d-cite> obtains the $\tilde{O}(H^3/d_m\epsilon^2)$ complexity for the local uniform OPE task and <d-cite key="yin2021optimal"></d-cite> improves the result to $\tilde{O}(H^2/d_m\epsilon^2)$ in the time-homogeneous setting, with model-based estimators. The key for the tight dependence is due to **the singleton absrobing MDP technique**, and importantly, such a technique is not confined to local uniform OPE task only and can be adapted to challening settings like offline task-agnostic learning and offline reward-free learning.



## The challenge in optimality for time-homogeneous RL  

For analyzing either learning or uniform evaluation tasks, at some point one needs to apply concentration inequalities (usually Bernstein inequality for the sharp result) over the term $(\widehat{P}-P)\widehat{V}^\pi_t$ for the current estimates. Espeically, in offline RL, those $\widehat{P}$ are usually constructed via the model-based estimates (*e.g.* <d-cite key="kidambi2020morel"></d-cite>). When the MDP is time-inhomogeneous, $$\widehat{P}_t$$ and $\widehat{V}^\pi_{t+1}$ are conditionally independent due to the construction

$$
  \widehat{P}_t(s^{\prime} \mid s, a)=\frac{\sum_{i=1}^n\mathbf{1}[(s^{(i)}_{t+1},a^{(i)}_t,s^{(i)}_t)=(s^\prime,s,a)]}{n_{s_t,a_t}}
$$

and $n_{s_t,a_t}=\sum_{i=1}^n \mathbf{1}[s_t^{(i)},a_t^{(i)}=s,a]$. To further kill the dependence in $H$, the time-inhomogeneous case denotes $n_{s, a}:=\sum_{i=1}^{n} \sum_{h=1}^{H} \mathbf{1}\left[s_{h}^{(i)}, a_{h}^{(i)}=s, a\right]$ and uses the estimate 

$$
\widehat{P}\left(s^{\prime} \mid s, a\right)=\frac{\sum_{i=1}^{n} \sum_{h=1}^{H} \mathbf{1}\left[\left(s_{h+1}^{(i)}, a_{h}^{(i)}, s_{h}^{(i)}\right)=\left(s^{\prime}, s, a\right)\right]}{n_{s, a}}
$$ 

In this scenario, $\widehat{P}$ and $\widehat{V}^\pi_{t+1}$ are no longer independent since $\widehat{P}$ uses the samples acorss different times. How to deal with it?



## Singleton absorbing MDP: A sharp analysis tool

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/Singleton_MDP_1.pdf">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/Singleton_MDP_3.pdf">
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/Singleton_MDP_2.pdf">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/Singleton_MDP_4.pdf">
    </div>
</div>
<div class="caption">
    The left panel shows the covering-based absorbing MDP, and the right panel shows the singleton absorbing MDP.
</div>


The visualization can help understand the *singleton absorbing MDP* technique. The upper part demonstrates the infinite horizon case and the bottom part demonstrates the finite horizon case. In particular, it should be a $H$-dimensional hypercube $[0,H]^H$ (that contains $\widehat{V}^\star_1,\ldots,\widehat{V}^\star_h$) instead of only the square $[0,H]\times[0,H]$ ($\widehat{V}^\star_1,\widehat{V}^\star_2$). This is only for the ease of visualization.

The standard absorbing MDP technique <d-cite key="agarwal2020model"></d-cite> leverages a set of absorbing MDPs to cover the range of value functions (following the standard covering principle) to make sure $\widehat{V}^\star$ is close to one of the element (absorbing MDP) in the set (left panel). The size of the covering set (*i.e.* the covering number) grows exponentially in $H$ in the finite horizon setting and this is due to the fact that there are $\widehat{V}^\star_1,\widehat{V}^\star_2,\ldots,\widehat{V}^\star_H$ quantities to cover. This results in the metric entropy to blow up by a factor of $H$ and incurs suboptimality. On the other hand, the singleton absorbing MDP $\widehat{V}^\star_{h,u^\star}$ can completely get rid of the covering issue, maintain the independence and  control the error propagation ($\lvert\lvert\widehat{V}^\star-\widehat{V}^\star_{u^\star}\rvert\rvert_\infty$ is sufficiently small). 

Now that for a state $s$ we can design $\widehat{V}^\star_{h,u^\star}$ such that $$\widehat{P}_s$$ is independent of $\widehat{V}^\star_{h,u^\star}$ and concentration inequalities apply! (check <d-cite key="yin2021optimal"></d-cite> for details)  



## Hightlights of our results

The model-based design together with the singleton absorbing MDP analysis is able to achieve the following:

<ul>
    <li>For finite horizon time-invariant setting, $\epsilon$-optimal local uniform OPE is guaranteed with complexity $\tilde{O}(H^2/d_m\epsilon^2)$;
   </li>
    <li>
    For finite horizon time-invariant setting, $\epsilon$-optimal policy is guaranteed with complexity $\tilde{O}(H^2\log(K)/d_m\epsilon^2)$ for the offline task-agnotic learning;
    </li>
    <li>
    	For finite horizon time-invariant setting, $\epsilon$-optimal policy is guaranteed with complexity $\tilde{O}(H^2S/d_m\epsilon^2)$ for the offline reward-free learning;
	</li>
</ul>
All of above have minimax rates in their respective settings! If you are interested, please check <d-cite key="yin2021optimal"></d-cite> for a reference. 



## One Take-Away

One side product is that the singleton absorbing MDP works in the infinite horizon setting as well! This means singleton absorbing MDP is not only optimal and but also agnostic to the settings.  

