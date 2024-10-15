---
layout: page
permalink: /research/
title: Research 
description: This page provides the summary of my current research. Check <a href='https://mingyin0312.github.io/publications/'>Publications</a> page for papers.
nav: true
---


***

<br>

#### **Theoretical Foundation for RL** 

<br>


<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/RL_theory.pdf' | relative_url }}" alt="" title="RL Foundation" style="width: 700px; height: auto;" />
    </div>
</div>
<div class="caption">
    
</div>

**Reinforcement learning** (RL) has rapidly become one of the fastest-growing fields in machine learning. Over the past decade, RL applications have achieved major breakthroughs, such as defeating world champions in Go and StarCraft II. However, the use of RL in real-world problems remains limited. The main challenge is that most RL methods require interaction with an environment, which is often infeasible in practice due to high costs, and potential legal, ethical, or safety concerns. My research in RL foundation aims to develop efficient RL algorithms that can learn from offline data with low (optimal) sample and computation complexity. There are two main tasks in these two categories: **Offline Policy Learning (OPL)**: learning the optimal strategy for sequential decision-making problems with the logged historical data; **Offline Policy Evaluation (OPE)**: making the counterfactual prediction for the performance of an undeployed strategy using historical data.




<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/RS1.pdf' | relative_url }}" alt="" title="OPL" style="width: 300px; height: auto;"/>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/OPE.pdf' | relative_url }}" alt="" title="OPE" style="width: 310px; height: auto;" />
    </div>
</div>
<div class="caption">
    
</div>

Additionally, I also provide provable guarantees for online reinforcement learning (RL), where exploration is permitted, as well as for low adaptive RL, which bridges the gap between online and offline RL. I approach these problems from various perspectives, including posterior sampling, adversarial robustness, zero-sum games, and bandit formulations.




***

<br>




#### **Practical Reinforcement Learning**
<br>



<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/mujoco.png' | relative_url }}" alt="" title="mujoco" style="width: 141px; height: auto;"/>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/atari.jpg' | relative_url }}" alt="" title="atari" style="width: 218px; height: auto;" />
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-" src="{{ '/assets/img/Research/networkgym.png' | relative_url }}" alt="" title="networkgym" style="width: 350px; height: auto;"/>
    </div>
</div>
<div class="caption">
    
</div>

To move from foundation to practice, I also study how to develop principled methodologies to make Reinforcement Learning effective for practical applications. My work has successfully applied RL algorithms to physics-based robotics simulators, computer games, and internet network management problems.

For multi-access network traffic management, our pessimism-based algorithm outperforms existing state-of-the-art deep RL methods in the high-fidelity  <a href="https://ersp.cs.ucsb.edu/2020-2021-projects/group-4-20202021">NetworkGym</a> environment, which simulates multiple network traffic flows and multi-access traffic splitting. In offline RL, we introduced the <a href="http://proceedings.mlr.press/v202/li23av/li23av.pdf">Closed-Form Policy Improvement Operator</a>, which updates policies using a closed-form solution rather than traditional gradient descent, and it has proven competitive with leading RL methods in MuJoCo robotic simulators. Additionally, we reformulated the value function learning problem as a <a href="https://assets.amazon.science/8d/22/9dcd6e48482d800c30c654194e51/learning-the-target-network-in-function-space.pdf">target network learning</a> problem, yielding empirical improvements on Atari games.


***

<br>

#### **LLM Alignment meets Sequential Decision-making: Theory and Practice**


<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/auto.pdf' | relative_url }}" alt="" title="mujoco" style="width: 320px; height: auto;" />
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/spec_new.pdf' | relative_url }}" alt="" title="atari" style="width: 350px; height: auto;" />
    </div>
</div>
<div class="caption">
    Standard Auto-Regressive Decoding (Left) vs. Speculative Decoding (Right)
</div>

<br>

The Theoretical understanding of Large Language models lags far behind its empirical successes. I make an effort by studying **speculative decoding**, a popular decoding method that has been deployed in many real-world products and achieves a 2-2.5X LLM inference speedup while preserving the quality of the outputs. I conceptualize the decoding problem via Markov chain abstraction and study the theoretical characterization of the key properties, output quality, and inference acceleration. <a href="http://">The analysis</a> covers the theoretical limits of speculative decoding, batch algorithms, and output quality-inference acceleration tradeoffs. It uncovers fundamental connections within LLMs through total variation distances, showing how these components interact to impact the efficiency of decoding algorithms.





<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/transfer_q.png' | relative_url }}" alt="" title="mujoco" style="width: 320px; height: auto;" />
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/cd_tq.jpg' | relative_url }}" alt="" title="atari" style="width: 250px; height: auto;" />
    </div>
</div>
<div class="caption">

</div>

<br>

**Alignment** for large language models (LLMs) is essential to ensure that these models behave in ways that are safe, ethical, and aligned with human values and intentions. Given a reward model and an LLM to be aligned, alignment can be cast as a (sparse) Reinforcement Learning problem.  Ideally, a principled decoding method would use the optimal Q* function, but this function is unknown. The prior method estimates Q* by Q-SFT, and our method estimates the correct Q* function via the <a href="https://arxiv.org/pdf/2405.20495">Transfer procedure</a>.   




***

<br>

#### **Evaluation & Benchmarks for Generative AI**

<br>


<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/MMMU1.png' | relative_url }}" alt="" title="mujoco" style="width: 720px; height: auto;" />
    </div>
</div>
<div class="caption">
MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
</div>

Measuring the performance of generative AI models is challenging since their outputs are subjective, open-ended, and lack clear ground truth, requiring evaluation across multiple dimensions such as relevance, coherence, and diversity. This becomes even more complex in multimodal evaluations, where tasks involve various formats such as images and text. In collaboration with other researchers, we created <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf">MMMU</a>, a benchmark designed to evaluate expert-level multimodal understanding and reasoning across six diverse disciplines and over 30 subjects. **My contribution:** I contributed 400 carefully curated multimodal questions to a total of 11.5K questions in the benchmark.


<!-- MMMU has been the go-to-evaluation Benchmark for evaluating model's multimodal capabilities, examples include Gemini, GPT-4o, Claude 3, and LLama 3.2.   -->



<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/theoremqa.png' | relative_url }}" alt="" title="mujoco" style="width: 720px; height: auto;" />
    </div>
    <!-- <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/MMMU2.png' | relative_url }}" alt="" title="atari"  />
    </div> -->
</div>
<div class="caption">
    TheoremQA: A Theorem-driven Question Answering Dataset
</div>

Recent large language models (LLMs), such as GPT-4, have made significant progress in solving fundamental math problems like those in GSM8K, achieving over 90% accuracy. However, their ability to tackle more complex problems that require domain-specific knowledge, such as applying theorems, remains underexplored. To address this, we developed <a href="https://aclanthology.org/2023.emnlp-main.489.pdf">TheoremQA</a>, the first theorem-driven question-answering dataset aimed at evaluating AI models' ability to apply theorems to solve advanced science problems. **My contribution:** I curated nearly 100 college-level math competition questions to a total of 800 questions in the dataset.





***

<br>

#### **AI for Science & Engineering**

<br>


<div class="row text-center">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/crispr.png' | relative_url }}" alt="" title="mujoco" style="width: 250px; height: auto;" />
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img" src="{{ '/assets/img/Research/CrisprGpt.png' | relative_url }}" alt="" title="atari" style="width: 450px; height: auto;" />
    </div>
</div>
<div class="caption">
    
</div>


Genome engineering technology has revolutionized biomedical research by enabling precise genetic modifications. One exciting example is the CRISPR technology, which can be used to develop critical advances in patient care or even cure lifelong inherited diseases. This technology won the 2020 Nobel Prize in Chemistry. In assisting beginner researchers with gene-editing from scratch, we designed CRISPR-GPT, an LLM agent system to automate and enhance the CRISPR-based gene-editing design process. This system is driven by multi-agent collaboration, and it incorporates domain expertise, retrieval techniques, and external tools. I contributed to CRISPR-GPT by fine-tuning a specialized LLM with a decade’s worth of open-forum discussions among gene-editing scientists.


<!-- However, designing effective gene-editing experiments requires a deep understanding of both the CRISPR technology and the biological system involved. Meanwhile, despite their versatility and promise, Large Language Models (LLMs) often lack domain-specific knowledge and struggle to accurately solve biological design problems. In this work, we present CRISPR-GPT, an LLM agent system to automate and enhance the CRISPR-based gene-editing design process. CRISPR-GPT leverages the reasoning capabilities of LLMs for complex task decomposition, decision-making, and interactive human-AI collaboration. This system is driven by multi-agent collaboration, and it incorporates domain expertise, retrieval techniques, external tools, and a specialized LLM fine-tuned with a decade’s worth of open-forum discussions among gene-editing scientists. CRISPR-GPT assists users in selecting CRISPR systems, experiment planning, designing gRNAs, choosing delivery methods, drafting protocols, designing assays, and analyzing data. We showcase the potential of CRISPR-GPT in assisting beginner researchers with gene-editing from scratch, knocking-out four genes with CRISPR-Cas12a in a human lung adenocarcinoma cell line and epigenetically activating two genes using CRISPR-dCas9 in human melanoma cell line, both successful on first attempt. CRISPR-GPT enabled fully AI-guided gene-editing experiment design across different modalities, validating its effectiveness as an AI co-pilot in genome engineering. -->


