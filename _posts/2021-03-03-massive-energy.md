---
layout: post
title:  Why can't we surpass the speed of light? Einstein tells you
date:   2021-03-03 
description: Only senior high school knowledge is needed to understand this pheonmenon!
---
Why can't we travel faster than the speed of light? Einstein explained this pheonmenon through his famous equation

\begin{equation}\label{eqn:mass}
m=\frac{m_0}{\sqrt{1-(\frac{v}{c})^2}},
\end{equation}

which is the last of his four celebrated works in 1905 (Albert Einstein's Year of Miracles) titled [Does the inertia of a body depend on its energy content?](https://einsteinpapers.press.princeton.edu/vol2-trans/186). This paper entails arguably the most famous formula so far: [Mass-Energy Equivalence](https://en.wikipedia.org/wiki/Mass–energy_equivalence) 

$$E=mc^2.$$

From \eqref{eqn:mass}, when the velocity of a particle approaches $$c$$, the relative mass $$m$$ will goes to $$\infty$$. Therefore any particle with $$m_0>0$$ can never exceed the speed of light. 

For the rest of the post, we use high school physics to proof these famous results!

<hr>

### 1. Review from high school physics materials
<ul>
    <li>Momentum Conservation Principle (动量守恒定律): For a system (with two objectives) has 0 momentum, it always holds:
    	\begin{equation}\label{eqn:mon}
  		0=P_1-P_2=m_1v_1-m_2v_2,
  		\end{equation}
    	multiply both sides by time t, we further have 
    	\begin{equation}\label{eqn:mon_s}
    	0=m_1S_1-m_2S_2.
    	\end{equation}
   </li>
    <li>
    Electromagnetic waves: 
    \begin{equation}\label{eqn:egw}
    E=Pc
    \end{equation}	
    </li>
    <li>
    	Theorem of Momentum:
    	\begin{equation}\label{eqn:tom}
    		dW=Fdx=dE,\quad F=\frac{dP}{dt}.
    	\end{equation}
	</li>
</ul>

<hr>

### 2. Proof of $$E=mc^2$$

**Einstein's Thought Experiment.** Imagine there is a box with length $$L$$, a photon with energy $$E$$ goes from one side to the other side. Consider the system of photon and the box.

By \eqref{eqn:egw}, 

\begin{equation}
P_1=\frac{E}{c}
\end{equation}

then by \eqref{eqn:mon},

\begin{equation}
P_2=P_1=\frac{E}{c}.
\end{equation}

Then by the definition of momentum \eqref{eqn:mon}

\begin{equation}
v_{\text{box}}=\frac{P_2}{m_2}=\frac{E}{m_2c}.
\end{equation}

On the other hand, for photon $$s_1\approx L$$, then for box we have 

\begin{equation}
s_2=v_{\text{box}}\cdot t=\frac{E}{m_2c}\cdot t=\frac{E}{m_2c}\cdot \frac{L}{c}
\end{equation}

Let the mass of photon be $$m$$, then by Momentum Conservation Principle, 

\begin{equation}
mL-m_2\cdot \frac{EL}{m_2c^2}=0
\end{equation}

Which gives 

$$E=mc^2.$$

This can be generalize to not only photon but all particles!

Quite easy! Right?

<hr>

### Proof of $$m={m_0}/{\sqrt{1-(\frac{v}{c})^2}}$$

Recall $$E=mc^2$$, $$P=mv$$, hence 

\begin{equation}\label{eqn:inter}
\frac{E}{P}=\frac{c^2}{v}\Rightarrow E=\frac{Pc^2}{v}.
\end{equation}

Now by \eqref{eqn:tom}, 

\begin{equation}
	dE=Fdx=\frac{dP}{dt}dx=v\cdot dP,
\end{equation}
multiply above by \eqref{eqn:inter}, then

$$
EdE=\frac{Pc^2}{v}\cdot v\cdot dP=Pc^2\cdot dP.
$$

Take integral, we obtian 

$$
E^2=c^2P^2+const.
$$

When $$P=0$$, $$E=E_0=m_0c^2$$. Therefore we already derived

\begin{equation}\label{eqn:final}
E^2=c^2P^2+m_0^2c^4
\end{equation}

In particular, for photon, by \eqref{eqn:egw} $$E=Pc$$, which implies the invariant mass (光子的静止质量) $$m_0=0$$! This simly says: **For any pariticle that has positive invariant mass, it cannot achieve the speed of light!**


Finally, use \eqref{eqn:inter}, we have $$P=\frac{Ev}{c^2}$$, plug in \eqref{eqn:final} to obtain

$$
E^2=E^2\frac{v^2}{c^2}+m_0^2c^4\Rightarrow E=\frac{m_0c^2}{\sqrt{1-\frac{v^2}{c^2}}}.
$$

Again, use $$E=mc^2$$, we obtain

$$
m=\frac{m_0}{\sqrt{1-(\frac{v}{c})^2}}.
$$

That's it! Now you know why $$c$$ is the fastest speed we can get!


