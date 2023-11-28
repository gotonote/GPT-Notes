第二章：Prompt-Tuning 的定义
---------------------

> 涉及知识点：
> 
> *   Template 与 Verbalizer 的定义；
>     

那么什么是 Prompt 呢？在了解预训练语言模型的基础，以及预训练语言模型在 Pre-training 和 Fine-tuning 之后，我们已经可以预想到 **Prompt 的目的是将 Fine-tuning 的下游任务目标转换为 Pre-training 的任务** 。那么具体如何工作呢？

我们依然以二分类的情感分析作为例子，描述 Prompt-tuning 的工作原理。给定一个句子`[CLS] I like the Disney films very much. [SEP]` 传统的 Fine-tuning 方法是将其通过 BERT 的 Transformer 获得 `[CLS]`表征之后再喂入新增加的 MLP 分类器进行二分类，预测该句子是积极的（positive）还是消极的（negative），因此需要一定量的训练数据来训练。

而 Prompt-Tuning 则执行如下步骤：

*   **构建模板（Template Construction）** ：通过人工定义、自动搜索、文本生成等方法，生成与给定句子相关的一个含有`[MASK]`标记的模板。例如`It was [MASK].`，并拼接到原始的文本中，获得 Prompt-Tuning 的输入：`[CLS] I like the Disney films very much. [SEP] It was [MASK]. [SEP]`。将其喂入 BERT 模型中，并复用预训练好的 MLM 分类器（在 huggingface 中为 BertForMaskedLM），即可直接得到`[MASK]`预测的各个 token 的概率分布；
    
*   **标签词映射（Label Word Verbalizer）** ：因为`[MASK]`部分我们只对部分词感兴趣，因此需要建立一个映射关系。例如如果`[MASK]`预测的词是 “great”，则认为是 positive 类，如果是 “terrible”，则认为是 negative 类。
    

> 此时会有读者思考，不同的句子应该有不同的 template 和 label word，没错，因为每个句子可能期望预测出来的 label word 都不同，因此如何最大化的寻找当前任务更加合适的 template 和 label word 是 Prompt-tuning 非常重要的挑战。

*   **训练** ：根据 Verbalizer，则可以获得指定 label word 的预测概率分布，并采用交叉信息熵进行训练。此时因为只对预训练好的 MLM head 进行微调，所以避免了过拟合问题
    

在 hugging face 上也可以直接进行测试：

*   I like the Disney films very much.
    
<div align=center>
<img src="./imgs/1.jpg" width="800" height="200">
</div>

*   I dislike the Disney films very much.
    
<div align=center>
<img src="./imgs/2.jpg" width="800" height="220">
</div>

其实我们可以理解，引入的模板和标签词本质上也属于一种数据增强，通过添加提示的方式引入先验知识
