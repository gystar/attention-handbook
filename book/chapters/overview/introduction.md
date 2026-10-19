# 1.1 从表示学习到 Attention 机制

语言模型生成一句话，需要根据已有文本判断接下来可能出现什么。但文本中的信息并不彼此独立：一个词的含义可能取决于邻近的修饰语，一处代词可能指向前文的人物，回答问题所需的线索也可能分散在多个段落中。

模型如何表示这些信息，又如何将不同位置的信息联系起来？本节从表示学习出发，介绍上下文建模的需求，再说明注意力机制（Attention Mechanism）在 Transformer 与大语言模型中的作用。

## 从数据中学习表示

机器学习（Machine Learning）利用数据学习完成任务所需的规律。以图像分类为例，模型接收图像并预测类别；训练根据预测与目标之间的差异调整参数，使模型能够处理未见过的图像。训练完成后，使用模型处理新输入的过程称为推理（Inference）。

这个过程也取决于输入如何表示。一张图像可以表示为像素数组，也可以用边缘、纹理和形状等特征描述。不同表示使任务中有用的信息呈现出不同形式。深度学习（Deep Learning）通过多层可学习的变换构建表示：模型不仅学习预测，也学习如何提取和组合中间信息。这是表示学习（Representation Learning）的一种实现方式。[Goodfellow et al., 2016，Chapter 1](https://www.deeplearningbook.org/contents/intro.html)

语言模型同样需要数值输入。Tokenizer 将文本转换为 Token 序列及整数编号；Token 可以对应一个词、词的一部分或其他文本单元。Embedding 再将编号映射为向量，作为网络计算的起点。这里的向量是一组数值，而不是直接存放在模型中的文字解释。[Hugging Face，Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4?fw=pt)

![法语句子中的三个词分别对应三个向量，向量以绿色方格表示](../../assets/overview/alammar-embeddings.png)

图 1-1a　文本的向量表示。绿色方格表示向量分量，不代表实际模型只有四个维度。引自 Jay Alammar，*The Illustrated Transformer*，原图未修改，CC BY-NC-SA 4.0。[原文](https://jalammar.github.io/illustrated-transformer/)；[许可](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

## 语言表示为什么需要上下文

初始 Token Embedding 通常由编号查表得到。同一个 Token 会取得相同的初始向量，但在不同文本中可能具有不同含义。因此，模型需要在后续计算中根据上下文更新表示；位置信息也需要通过相应机制参与计算。

3Blue1Brown 的教程以英文词 *mole* 为例：它在不同短语中可以表示鼹鼠、物质的量单位“摩尔”或皮肤上的痣。周围词语帮助确定当前语义。这个例子将词视为 Token 以便教学，实际切分仍取决于 Tokenizer。[3Blue1Brown，Motivating Examples](https://www.3blue1brown.com/lessons/attention/)

上下文的作用也不局限于词义消歧。例如，“这杯茶太烫了，我想等它凉一些再喝”中的“它”，需要结合前面的“茶”理解。模型需要形成的表示因此不只是“当前位置是什么 Token”，还包括它与当前文本中其他信息的联系。

这构成了序列建模的一个基本要求：不同位置必须能够交换信息。Recurrent Neural Network（RNN）通过沿序列更新的状态传递信息；Attention 机制则提供了按匹配关系汇总不同位置信息的方式。两者可以共同使用，历史演进将在后文讨论。[Vaswani et al., 2017，§1—3](https://arxiv.org/html/1706.03762v7)

## Attention 机制如何汇总信息

Attention 机制根据当前输入计算权重，并组合相应的向量。对于本书首先讨论的 Softmax Attention，可以依次理解为三个动作：比较当前位置与可访问位置的表示，将匹配分数归一化，再按权重汇总信息。正式的 Query、Key、Value 定义与计算公式留到下一节。

![原教程中 it 所在位置与其他输入位置之间的 Attention 可视化，部分连线指向 The 和 animal](../../assets/overview/alammar-self-attention.png)

图 1-1b　原教程中的 Self-Attention 可视化。右侧选中的是 `it`，左侧展示参与关联的输入位置；该图来自 Encoder 场景，不受自回归模型只能访问自身及前文的限制。引自 Jay Alammar，*The Illustrated Transformer*，原图未修改，CC BY-NC-SA 4.0。[原文及可视化背景](https://jalammar.github.io/illustrated-transformer/)；[许可](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

图中的信息关联帮助解释 Attention 层的作用：它输出的是结合上下文的向量，而不是直接输出一条指代判断。一个 Head 的权重只描述特定层、特定输入下的一次信息汇总；模型的最终输出还取决于其他 Head、后续层与输出计算。

这种汇总是动态的。模型训练得到的投影等参数在常规推理中保持固定，但更换输入或当前位置后，匹配分数和权重会随之变化。区分“训练得到的参数”与“根据输入计算的权重”，是理解 Attention 机制的关键。[Shazeer, 2019，§2.1—2.3](https://arxiv.org/html/1911.02150v1)

## Attention 机制在 Transformer 与 LLM 中的位置

Attention 是一种计算机制，Transformer 是组织这些计算的网络架构。Self-Attention 在同一序列的位置之间汇总信息，前馈网络（Feed-Forward Network，FFN）则对各位置的表示分别进行变换。残差连接与归一化等组件也参与层内计算；多层网络进一步更新表示。[Vaswani et al., 2017，§3.1—3.3](https://arxiv.org/html/1706.03762v7)

![两个位置的向量先经过共同的 Self-Attention 模块，再分别经过 Feed Forward Neural Network，输出传入下一层 Encoder](../../assets/overview/alammar-encoder.png)

图 1-1c　Encoder 层内的简化数据流。先观察 Self-Attention 的跨位置计算，再观察 FFN 的逐位置变换；图中省略了残差连接与归一化。该图用于定位组件，不是 Decoder-only LLM 的完整架构。引自 Jay Alammar，*The Illustrated Transformer*，原图未修改，CC BY-NC-SA 4.0。[原文](https://jalammar.github.io/illustrated-transformer/)；[许可](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

本书主要讨论基于 Transformer 的自回归大语言模型（Large Language Model，LLM）。生成时，模型根据已有文本计算下一个 Token 的概率，生成策略据此选择一个 Token，将其加入上下文，再继续预测。Decoder-only 架构直接处理这一序列，无需另设用于编码输入序列的 Encoder。

在这种架构中，Causal Self-Attention 只允许当前位置访问自身及此前位置；最后一个位置的最终表示用于计算下一 Token 的概率。Attention 层参与上下文信息汇总，输出层负责将最终表示转换为预测分数，二者承担不同作用。[Shazeer, 2019，§2.4](https://arxiv.org/html/1911.02150v1)；[Vaswani et al., 2017，§3.4](https://arxiv.org/html/1706.03762v7)

这些概念分别描述不同层面：深度学习是一类机器学习方法，Transformer 是网络架构，Attention 是计算机制，LLM 是模型类别。LLM 不规定唯一架构，Attention 机制也可以用于其他模型和任务。

随着上下文变长，需要处理的位置关系增多；逐步生成还涉及历史计算结果的保存与复用。后续章节将分析这些成本，并说明不同方法如何调整信息表示、连接范围、计算实现和缓存管理。

延伸观看：[3Blue1Brown，Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)。可先配合[作者图文的 Recap 与 Motivating Examples](https://www.3blue1brown.com/lessons/attention/)理解初始表示如何结合上下文，再观看 QKV 部分。本文保留必要解释，视频不作为阅读后续内容的前提。

下一节：[1.2 Attention 机制的基本计算](attention-in-llms.md)。
