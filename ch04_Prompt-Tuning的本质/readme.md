第四章：Prompt-Tuning 的本质
---------------------

> 涉及知识点：
> 
> *   元学习与 prompt；
>     
> *   基于 Prompt 的 NLP 任务的统一范式；
>     
> *   基于生成模型的 Prompt；
>     
> *   Prompt 与参数有效性学习；
>     

前面章节介绍了大量与 Prompt 相关的内容，我们可以发现，最初的 Prompt Tuning 是旨在设计 Template 和 Verbalizer（即 Pattern-Verbalizer Pair）来解决基于预训练模型的小样本文本分类，然而事实上，NLP 领域涉及到很多除了分类以外其他大量复杂的任务，例如抽取、问答、生成、翻译等。这些任务都有独特的任务特性，并不是简单的 PVP 就可以解决的，因而， **我们需要提炼出 Prompt Tuning 的本质，将 Prompt Tuning 升华到一种更加通用的范式上** 。

博主根据对 Prompt-Tuning 两年多的研究经验，总结了三个关于 Prompt 的本质，如下：

*   Prompt 的本质是一种对任务的指令；
    
*   Prompt 的本质是一种对预训练任务的复用；
    
*   Prompt 的本质是一种参数有效性学习；
    