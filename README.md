<div align="center">
  <h1 style="border-bottom: none;">Rethinking Infrared Small Target Detection: A Foundation-Driven Efficient Paradigm</h1>
</div>

<div align="center">
  <p>
    <a href="https://scholar.google.com/citations?user=Dd4_VW8AAAAJ&hl=zh-CN" target="_blank">Chuang Yu</a><sup>1,2,6</sup>,&nbsp
    <a href="https://scholar.google.com/citations?user=3cBa6r4AAAAJ&hl=zh-CN" target="_blank">Jinmiao Zhao</a><sup>1,2</sup>,&nbsp
    <a>Yunpeng Liu</a><sup>1*</sup>,&nbsp
    <a href="https://iron-lyk.github.io/" target="_blank">Yaokun Li</a><sup>3</sup>,&nbsp
    <a>Xiujun Shu</a><sup>4</sup>,
    <br>
    <a>Yuanhao Feng</a><sup>4</sup>,&nbsp
    <a>Bo Wang</a><sup>4</sup>,&nbsp
    <a href="https://scholar.google.com/citations?user=y5Ov6VAAAAAJ&hl=zh-CN" target="_blank">Yimian Dai</a><sup>5</sup>,&nbsp
    <a href="https://scholar.google.com/citations?user=-xQ-C1sAAAAJ&hl=zh-CN" target="_blank">Xiangyu Yue</a><sup>6*</sup>
  </p>
  <p>
    <sup>1</sup> Shenyang Institute of Automation, Chinese Academy of Sciences
     <br>
    <sup>2</sup> University of Chinese Academy of Sciences &nbsp;&nbsp;
     <br>
    <sup>3</sup> Sun Yat-sen University &nbsp;&nbsp;
    <sup>4</sup> Tencent &nbsp;&nbsp;
    <sup>5</sup> Nankai University &nbsp;&nbsp;
    <sup>6</sup> MMLab, The Chinese University of Hong Kong
  </p>
</div>

## 💥 Abstract
While large-scale visual foundation models (VFMs) exhibit strong generalization across diverse visual domains, their potential for single-frame infrared small target (SIRST) detection remains largely unexplored. **To fill this gap, we systematically introduce the frozen representations from VFMs into the SIRST task for the first time and propose a Foundation-Driven Efficient Paradigm (FDEP), which can seamlessly adapt to existing encoder-decoder-based methods and significantly improve accuracy without additional inference overhead.** Specifically, a Semantic Alignment Modulation Fusion (SAMF) module is designed to achieve dynamic alignment and deep fusion of the global semantic priors from VFMs with task-specific features. Meanwhile, to avoid the inference time burden introduced by VFMs, we propose a Collaborative Optimization-based Implicit Self-Distillation (CO-ISD) strategy, which enables implicit semantic transfer between the main and lightweight branches through parameter sharing and synchronized backpropagation. In addition, to unify the fragmented evaluation system, we construct a Holistic SIRST Evaluation (HSE) metric that performs multi-threshold integral evaluation at both pixel-level confidence and target-level robustness, providing a stable and comprehensive basis for fair model comparison. Extensive experiments demonstrate that the SIRST detection networks equipped with our FDEP framework achieve state-of-the-art (SOTA) performance on multiple public datasets. 


## 🚀 Overview
<div align="center">
<br>
  <img width="100%" src="imgs/FDEP-Framework-Overview.png" alt="FDEP-Framework-Overview">
<br>
</div>

## ✅ TODO List
We are finalizing the release of the paper, dataset and code and aim to complete it as soon as possible. Please stay tuned! ⚡⚡⚡
- [ ] Release paper.
- [ ] Release training and inference code.









