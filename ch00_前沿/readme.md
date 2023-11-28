# 前沿

## 一、预训练语言模型的发展历程

<div align=center>
<img src="./imgs/1.jpg" width="800" height="280">
</div>
<div align=center>图1. 预训练语言模型的发展历程</div>

截止 23 年 3 月底，语言模型发展走过了三个阶段：

*   **第一阶段** ：设计一系列的自监督训练目标（MLM、NSP 等），设计新颖的模型架构（Transformer），遵循 Pre-training 和 Fine-tuning 范式。典型代表是 BERT、GPT、XLNet 等；
    
*   **第二阶段** ：逐步扩大模型参数和训练语料规模，探索不同类型的架构。典型代表是 BART、T5、GPT-3 等；
    
*   **第三阶段** ：进入AIGC（Artificial Intelligent Generated Content）时代，模型参数规模达到千万亿级别，模型架构采用自回归方法，大型模型逐渐发展为对话式、生成式和多模态模型。这一阶段的模型更加注重与人类交互的对齐，以实现可靠、安全且无毒的模型。典型的代表模型有InstructionGPT、ChatGPT、Bard和GPT-4等。

## 二、面向预训练语言模型的 Prompt-Tuning 技术发展历程

<div align=center>
<img src="./imgs/2.jpg" width="800" height="280">
</div>
<div align=center>图2. Prompt-Tuning 技术发展历程</div>

Prompt-Tuning 自从 GPT-3 被提出以来，从传统的离散、连续的 Prompt 的构建、走向面向超大规模模型的 In-Context Learning、Instruction-tuning 和 Chain-of-Thought。

自从 GPT、EMLO、BERT 的相继提出，以`Pre-training + Fine-tuning` 的模式在诸多自然语言处理（NLP）任务中被广泛使用，其先在`Pre-training`阶段通过一个模型在大规模无监督语料上预先训练一个 **预训练语言模型（Pre-trained Language Model，PLM）** ，然后在`Fine-tuning`阶段基于训练好的语言模型在具体的下游任务上再次进行 **微调（Fine-tuning）** ，以获得适应下游任务的模型。

这种模式在诸多任务的表现上超越了传统的监督学习方法，不论在工业生产、科研创新还是竞赛中均作为新的主流方式。然而，这套模式也存在着一些问题。例如，在大多数的下游任务微调时， **下游任务的目标与预训练的目标差距过大** 导致提升效果不明显， **微调过程中依赖大量的监督语料** 等。

**至此，以 GPT-3、PET 为首提出一种基于预训练语言模型的新的微调范式——Prompt-Tuning** ，其旨在通过添加模板的方法来避免引入额外的参数，从而让语言模型可以在小样本（Few-shot）或零样本（Zero-shot）场景下达到理想的效果。Prompt-Tuning 又可以称为 Prompt、Prompting、Prompt-based Fine-tuning 等。

因此简单的来说，Prompt-Tuning 的动机旨在解决目前传统 Fine-tuning 的两个痛点问题：

*   **降低语义差异（Bridge the gap between Pre-training and Fine-tuning）** ：预训练任务主要以 Masked Language Modeling（MLM）为主，而下游任务则重新引入新的训练参数，因此两个阶段的目标通常有较大差异。因此需要解决如何缩小 Pre-training 和 Fine-tuning 两个阶段目标差距过大的问题；
    
*   **避免过拟合（Overfitting of the head）** ：由于在 Fine-tuning 阶段需要新引入额外的参数以适配相应的任务需要，因此在样本数量有限的情况容易发生过拟合，降低了模型的泛化能力。因此需要面对预训练语言模型的过拟合问题。
  
    

