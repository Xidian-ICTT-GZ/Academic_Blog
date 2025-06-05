# 大语言模型在代码生成相关方向的调研
****
## **引言**
### 调研背景
**1.对于自动生成代码的需求**：
随着数字化进程的加速，软件应用场景不断拓展，从传统的桌面软件、Web 应用，延伸至移动互联网、人工智能、物联网、区块链等新兴领域。软件规模和复杂度呈指数级增长，代码量大幅攀升，功能需求日益复杂，传统的手工编写代码方式效率低下，难以满足快速迭代的市场需求。
人工智能技术的飞速发展为代码生成带来了新的机遇。

**2.人工智能算法的发展和计算能力的提升**：
机器学习、深度学习算法的不断演进，使得计算机能够从大量数据中学习模式和规律。在代码生成领域，神经网络模型的应用逐渐成熟，从早期简单的基于规则的代码生成，发展到基于统计和机器学习的方法，再到如今强大的大语言模型（LLMs）。大语言模型基于 Transformer 架构，通过在大规模文本数据上进行预训练，能够学习到丰富的语言知识和语义表示。这种强大的学习能力使得大语言模型在代码生成任务中展现出巨大潜力，为解决软件开发中的效率和质量问题提供了新的途径。
随着互联网的普及和数字化进程的推进，产生了海量的代码数据。开源代码库如 GitHub、GitLab 等积累了数以亿计的代码文件，涵盖了各种编程语言、应用领域和开发风格。这些丰富的数据为大语言模型的训练提供了充足的素材，使得模型能够学习到多样化的代码模式和编程习惯。与此同时，计算硬件的不断升级，如 GPU（图形处理器）和 TPU（张量处理单元）的出现，大幅提升了计算能力。高性能计算集群的广泛应用，使得大语言模型的训练能够在更短的时间内完成，并且可以处理更大规模的数据。强大的计算能力为大语言模型在代码生成领域的发展提供了坚实的支撑。

**3.大语言模型生成代码的特点**：
早期的代码生成技术主要基于模板和规则，通过编写特定的代码模板，根据输入参数生成相应的代码片段。这种方式虽然在一定程度上提高了代码编写效率，但灵活性和通用性较差，难以适应复杂多变的需求。
大语言模型的出现，为代码生成带来了革命性的变化。大语言模型能够理解自然语言描述的编程需求，并生成相应的代码。它们不仅可以生成简单的代码片段，还能处理复杂的编程任务，如函数定义、类实现、算法设计等。在实际应用中，大语言模型已经在代码自动补全、代码生成辅助工具、智能编程助手等方面取得了显著成果，但也面临着一些挑战，如生成代码的准确性、安全性和可解释性等问题。
由于大语言模型是基于大规模数据的统计学习，其生成的代码可能会出现与实际需求不完全匹配的情况。模型可能生成语法正确但逻辑错误的代码，或者生成的代码虽然在常见场景下可行，但在一些特殊边界条件下无法正常工作。其次，大语言模型对于复杂编程概念和领域特定知识的理解可能不够准确和深入，导致生成的代码在专业性和高效性方面存在不足。对大语言模型生成代码不确定性的深入理解和有效控制，有助于推动代码生成技术的进一步发展，使其在更广泛的场景中得到应用，促进软件开发行业的创新与变革。
### 调研目的
**全面了解现状**：系统梳理大语言模型在代码生成相关领域的研究成果，总结现有研究的主要内容、方法和应用，构建多维度分类体系，清晰呈现该领域的研究现状与发展脉络，为后续研究提供基础。

**分析大语言模型的角色及作用**：深入分析 LLMs 在算法设计中的不同角色、各类搜索方法、提示策略以及广泛的应用领域，明确它们的优势、局限和适用场景，为进一步优化和应用 LLMs 提供参考。

**揭示问题挑战**：了解当前当前大语言模型在代码生成方向研究存在的问题和挑战，如可扩展性、可解释性、安全性、成本和创新等方面的不足，为研究人员提供改进方向，推动该领域的技术发展。

**探索未来方向**：基于现状分析和问题揭示，提出具有潜力的未来研究方向，包括开发领域特定的 LLMs、探索多模态 LLMs、促进人机协作等，为后续研究提供思路和指引，助力 LLMs 在算法设计领域的持续创新与发展。
****
## 主题一：基于大语言模型的算法设计与代码生成技术
算法设计在众多领域至关重要，传统设计方式需耗费大量人力且对专业知识要求高。随着人工智能发展，LLMs 凭借其大规模、训练充分和性能优越的特点，在数学推理、代码生成等领域取得显著进展。
过去三年，LLM4AD (LLMs for Algorithm Design)成为极具潜力的研究领域，能优化算法设计、提升效率并减少人力投入。然而，该领域缺乏系统综述，现有文献多聚焦于特定算法情境或领域应用。因此，对 LLM4AD 进行全面系统的调研十分必要。
#### 相关论文一：[A survey on large language models for code generation](https://arxiv.org/abs/2406.00515)
文章是关于代码生成大型语言模型（Code LLMs）的综述，主要介绍其在代码生成领域的研究进展、应用及挑战。文中首先梳理了 Code LLMs 的发展历程，涵盖数据处理、模型架构、训练方法（如预训练、指令微调、强化学习）、评估基准（如 HumanEval、MBPP、BigCodeBench）及实际应用（如 GitHub Copilot、CodeGeeX）。同时指出当前面临的挑战，如复杂代码生成能力不足、数据质量与多样性问题、评估体系不完善等，并展望了未来研究方向，如模型架构创新、多语言支持、安全对齐等。

##### 方向一：代码大语言模型原理

首先，文章提出了代码大语言模型的有效性从根本上归功于其庞大的模型参数量、大规模多样化数据集以及训练过程中投入的巨大算力的观点。通常而言，扩大语言模型规模能持续提升其在各类下游任务中的表现和样本效率。文章经过统计发现，当模型规模扩展到一定程度时（例如1750亿参数的GPT-3和5400亿参数的PaLM），大语言模型会展现出被称为"涌现能力"的不可预测现象，包括指令跟随、上下文学习和分步推理等——这些能力在小模型中不存在，却在大模型中显著显现。
用于代码生成的大型语言模型（LLM）指的是使用LLM从自然语言描述生成源代码，这一过程也被称为自然语言到代码的任务。通常，这些自然语言描述包括编程问题陈述（或文档字符串），并且可以选择性地包含一些编程上下文（例如，函数签名、断言等）。形式上，这些自然语言（NL）描述可以表示为$x$
。给定$$x$，使用具有模型参数$𝜃$的LLM来生成代码解决方案$y$可以表示为$𝑃_𝜃 (y|x)$。LLM中上下文学习能力的出现导致了将范例附加到自然语言描述$x$作为演示，以提高代码生成性能或约束生成格式。一个固定的𝑀个范例的集合表示为$\{(xi, yi)\}^M_{𝑖=1}$,因此，对于具有少量样本（或零样本）范例的代码生成，LLM的更通用公式可以修改为：

$$
\begin{equation}
\begin{aligned}
   P_\theta(\mathbf{y}\mid\mathbf{x}) \Rightarrow P_\theta(\mathbf{y}\mid\operatorname{prompt}(\mathbf{x}, \{(\mathbf{x_i}, \mathbf{y_i})\}_{i=1}^k)), k\in\{0, 1, \dots, M\}
\end{aligned}
\end{equation}
$$

大语言模型中的每个Transformer层都包含**多重注意力(Multi-Head Self-Attention)** 机制以识别一个token序列内在的语义关系，这个序列可以跨越不同的潜在表示空间。形式上，Transformer采用的MHSA机制可以表示如下：

$$
\begin{equation}
\begin{aligned}
    \mathbf{h}^{(l)}=\operatorname{MultiHeadSelfAttn}(\mathbf{Q},\mathbf{K},\mathbf{V}) =\operatorname{Concat}\left\{\mathrm{Head}_i\right\}_{i=1}^h\mathbf{W^O},
\end{aligned} 
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
    \operatorname{Head}_i =\operatorname{Attention}(\underbrace{\mathbf{H}^{(l-1)}\mathbf{W}_i^\mathbf{Q}}_{\mathbf{Q}},\underbrace{\mathbf{H}^{(l-1)}\mathbf{W}_i^\mathbf{K}}_{\mathbf{K}}, \underbrace{\mathbf{H}^{(l-1)}\mathbf{W}_i^\mathbf{V}}_\mathbf{V}), 
\end{aligned} 
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
    \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_{model}/h}}\right)\mathbf{V},
\end{aligned}
\end{equation}
$$

其中$H^{(𝑙−1)} ∈ R^{𝑛×𝑑_{𝑚𝑜𝑑𝑒𝑙}}$表示第$𝑙$个Transformer层的输入，而$h^{(𝑙)}∈R^{𝑛×𝑑_{𝑚𝑜𝑑𝑒𝑙}}$表示MHSA子层的输出。不同注意力头的数量由$ℎ$表示，而$𝑑_{𝑚𝑜𝑑𝑒𝑙}$指的是模型维度。投影集$\{W_𝑖^Q, W_𝑖^K, W_𝑖^V, W_𝑖^O\}∈
R^{𝑑_{𝑚𝑜𝑑𝑒𝑙}×𝑑_{𝑚𝑜𝑑𝑒𝑙}/ℎ}$包含了每个注意力头$Head_𝑖$的仿射变换参数，用于变换查询$Q$、键$K$、值$V$以及注意力子层的输出。$softmax$函数以逐行方式应用。查询和键的点积除以一个缩放因子$\surd d_{model}/h$以抵消过大的内积的潜在风险，并相应地减少$softmax$函数中的梯度，从而鼓励更平衡的注意力分布。此外，文章还介绍了掩码多头自注意力与交叉层多头自注意力机制。

在每个 Transformer 层中，在 MHSA 子层之后利用逐位置前馈网络 (PFFN) 以独立且相同的方式细化每个位置$𝑖$的序列嵌入，从而编码更复杂的特征表示。PFFN 由一对线性变换组成，中间穿插一个 ReLU 激活函数。形式上：
$$
\begin{equation}
\begin{aligned}
\operatorname{PFFN}(h^{(l)})=\left(\operatorname{Concat}\left\{\operatorname{FFN}(h^{(l)}_i)^T\right\}_{i=1}^{n}\right)^T,
\end{aligned}   
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
\operatorname{FFN}(h^{(l)}_i)=\operatorname{ReLU}(h^{(l)}_i\mathbf{W}^{(1)}+b^{(1)})\mathbf{W}^{(2)}+b^{(2)},
\end{aligned} 
\end{equation}
$$
其中$ℎ^{(𝑙)}∈R^{𝑛×𝑑_{𝑚𝑜𝑑𝑒𝑙}}$是第$l$个Transformer层中MHSA子层的输出，而$ℎ_𝑖^{(𝑙)} ∈ R^{𝑑_{𝑚𝑜𝑑𝑒𝑙}}$表示每个序列位置的潜在表示。投影矩阵$\{W^{(1)}, (W^{(2)})^𝑇\}∈R^{𝑑_{𝑚𝑜𝑑𝑒𝑙}×4𝑑_{𝑚𝑜𝑑𝑒𝑙}}$ 和偏置向量 ${b^{(1)}, b^{(2)}}∈R^{𝑑_{𝑚𝑜𝑑𝑒𝑙}}$是在训练期间学习的参数。这些参数在所有位置保持一致，同时从一层到另一层单独初始化。在此上下文中，$𝑇$表示矩阵的转置运算。

为了缓解因网络加深而导致的梯度消失或爆炸问题，Transformer模型在上述每个模块周围都加入了残差连接，然后进行层归一化。关于层归一化操作的位置，有两种广泛使用的方法：
1. 后归一化（Post-Norm）：层归一化在逐元素残差加法之后进行，与原始Transformer一致。
2. 预归一化（Pre-Norm）：层归一化应用于每个子层的输入，如GPT-2等模型。形式上，它可以表述为：
$$
\begin{equation}
\begin{aligned}
\textbf{Post-Norm}: 
    \mathbf{H^{(l)}} &=\operatorname{LayerNorm}(\operatorname{PFFN}(\mathbf{h^{(l)}})+\mathbf{h^{(l)}}),\\
    \mathbf{h^{(l)}}&=\operatorname{LayerNorm}(\operatorname{MHSA}(\mathbf{H^{(l-1)}})+\mathbf{H^{(l-1)}}) 
\end{aligned}
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
\textbf{Pre-Norm}: 
    \mathbf{H^{(l)}} &=\operatorname{PFFN}(\operatorname{LayerNorm}(\mathbf{h^{(l)}}))+\mathbf{h^{(l)}},\\
    \mathbf{h^{(l)}}&=\operatorname{MHSA}(\operatorname{LayerNorm}(\mathbf{H^{(l-1)}}))+\mathbf{H^{(l-1)}} 
\end{aligned}
\end{equation}
$$

近年来，大型语言模型（LLM）的快速发展导致大量模型通过持续预训练或微调被重新用于代码生成任务。这种趋势在开源模型领域尤为明显。例如，Meta AI 最初发布了 LLaMA 模型，随后发布了专门用于代码生成的 Code Llama。同样，DeepSeek 开发并发布的 DeepSeek LLM已扩展为 DeepSeek Coder，这是一个专门针对代码生成的变体。Qwen 团队在其原始 Qwen 模型的基础上开发并发布了 Code Qwen。另一方面，微软发布了 WizardLM，并正在探索其面向代码的对应版本 WizardCoder。谷歌也加入了这一行列，发布了 Gemma ，随后发布了 Code Gemma。除了简单地将通用 LLM 适应代码相关任务之外，专门为代码生成而设计的模型也大量涌现。值得注意的例子包括 StarCoder、OctoCoder 和 CodeGen。这些模型突出了 LLM 正在朝着代码生成方向发展这一趋势。
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_01.png" alt="Fig.1-1.1" style="display: block; margin: 0 auto;"/>
<p style="text-align: center;">Fig.1-1.1 用于代码生成的大型语言模型（LLMs）时序概览</p>

##### 方向二：代码大语言模型的架构
代码生成任务存在两种Transformer架构，包括编码器-解码器和仅解码器。对于编码器-解码器架构，它由编码器和解码器组成，其中编码器处理输入数据并生成一组表示，然后解码器使用这些表示来生成输出。然而，对于仅解码器架构，它仅由transformer的解码器部分组成，其中它使用单个层堆栈来处理输入数据并生成输出。因此，编码器-解码器架构适用于需要不同输入和输出域之间映射的任务，而仅解码器架构则设计用于专注于序列生成和延续的任务。具有这两种架构的LLM的概述如Fig.1-1.2所示。
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_02.png" alt="Fig.1-1.2"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-1.2 采用编码器-解码器和仅解码器进行代码生成的Transformer架构</p>

<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; border: none;">
  <thead>
    <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
      <th style="text-align: center; border: none;">Model</th>
      <th style="text-align: center; border: none;">Institution</th>
      <th style="text-align: center; border: none;">Size (Parameters)</th>
      <th style="text-align: center; border: none;">Vocabulary</th>
      <th style="text-align: center; border: none;">Context Window</th>
      <th style="text-align: center; border: none;">Date</th>
      <th style="text-align: center; border: none;">Open Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="text-align: center; border: none;">PyMT5</td>
      <td style="text-align: center; border: none;">Microsoft</td>
      <td style="text-align: center; border: none;">374M</td>
      <td style="text-align: center; border: none;">50K</td>
      <td style="text-align: center; border: none;">1024+1024</td>
      <td style="text-align: center; border: none;">2020-10</td>
      <td style="text-align: center; border: none;"></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">PLBART</td>
      <td style="text-align: center; border: none;">UCLA</td>
      <td style="text-align: center; border: none;">140M</td>
      <td style="text-align: center; border: none;">50K</td>
      <td style="text-align: center; border: none;">1024+1024</td>
      <td style="text-align: center; border: none;">2021-03</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">CodeT5</td>
      <td style="text-align: center; border: none;">Salesforce</td>
      <td style="text-align: center; border: none;">60M, 220M, 770M</td>
      <td style="text-align: center; border: none;">32K</td>
      <td style="text-align: center; border: none;">512+256</td>
      <td style="text-align: center; border: none;">2021-09</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">JuPyT5</td>
      <td style="text-align: center; border: none;">Microsoft</td>
      <td style="text-align: center; border: none;">350M</td>
      <td style="text-align: center; border: none;">50K</td>
      <td style="text-align: center; border: none;">1024+1024</td>
      <td style="text-align: center; border: none;">2022-01</td>
      <td style="text-align: center; border: none;"></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">AlphaCode</td>
      <td style="text-align: center; border: none;">DeepMind</td>
      <td style="text-align: center; border: none;">284M, 1.1B, 2.8B, 8.7B, 41.1B</td>
      <td style="text-align: center; border: none;">8K</td>
      <td style="text-align: center; border: none;">1536+768</td>
      <td style="text-align: center; border: none;">2022-02</td>
      <td style="text-align: center; border: none;"></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">CodeRL</td>
      <td style="text-align: center; border: none;">Salesforce</td>
      <td style="text-align: center; border: none;">770M</td>
      <td style="text-align: center; border: none;">32K</td>
      <td style="text-align: center; border: none;">512+256</td>
      <td style="text-align: center; border: none;">2022-06</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">ERNIE-Code</td>
      <td style="text-align: center; border: none;">Baidu</td>
      <td style="text-align: center; border: none;">560M</td>
      <td style="text-align: center; border: none;">250K</td>
      <td style="text-align: center; border: none;">1024+1024</td>
      <td style="text-align: center; border: none;">2022-12</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">PPOCoder</td>
      <td style="text-align: center; border: none;">Virginia Tech</td>
      <td style="text-align: center; border: none;">770M</td>
      <td style="text-align: center; border: none;">32K</td>
      <td style="text-align: center; border: none;">512+256</td>
      <td style="text-align: center; border: none;">2023-01</td>
      <td style="text-align: center; border: none;"></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">CodeT5+</td>
      <td style="text-align: center; border: none;">Salesforce</td>
      <td style="text-align: center; border: none;">220M, 770M, 2B, 6B, 16B</td>
      <td style="text-align: center; border: none;">50K</td>
      <td style="text-align: center; border: none;">2048+2048</td>
      <td style="text-align: center; border: none;">2023-05</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">CodeFusion</td>
      <td style="text-align: center; border: none;">Microsoft</td>
      <td style="text-align: center; border: none;">75M</td>
      <td style="text-align: center; border: none;">32K</td>
      <td style="text-align: center; border: none;">128+128</td>
      <td style="text-align: center; border: none;">2023-10</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
    <tr style="border-bottom: 1px solid #333;">
      <td style="text-align: center; border: none;">AST-T5</td>
      <td style="text-align: center; border: none;">UC Berkeley</td>
      <td style="text-align: center; border: none;">226M</td>
      <td style="text-align: center; border: none;">32K</td>
      <td style="text-align: center; border: none;">512+200/300</td>
      <td style="text-align: center; border: none;">2024-01</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
  </tbody>
</table>
<p style="text-align: center;">Table.1 采用编码器-解码器架构的大语言模型（LLMs）</p>

<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; border: none;">
  <thead>
    <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
      <th style="text-align: center; border: none;">Model</th>
      <th style="text-align: center; border: none;">Institution</th>
      <th style="text-align: center; border: none;">Size (Parameters)</th>
      <th style="text-align: center; border: none;">Vocabulary</th>
      <th style="text-align: center; border: none;">Context Window</th>
      <th style="text-align: center; border: none;">Date</th>
      <th style="text-align: center; border: none;">Open Source</th>
    </tr>
  </tdead>
<tbody>
<tr>
      <th style="text-align: center; border: none;">GPT-C  </td>
      <td style="text-align: center; border: none;">Microsoft </td>
      <td style="text-align: center; border: none;">366M      </td>
      <td style="text-align: center; border: none;">60K	</td>
      <td style="text-align: center; border: none;">1024	 </td>
      <td style="text-align: center; border: none;">2020-05 	 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">CodeGPT    </td>
      <td style="text-align: center; border: none;">Microsoft </td>
      <td style="text-align: center; border: none;">124M      	</td>
      <td style="text-align: center; border: none;">50K </td>
      <td style="text-align: center; border: none;">1024	 </td>
      <td style="text-align: center; border: none;">2021-02 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">GPT-Neo </td>
      <td style="text-align: center; border: none;">EleutderAI </td>
      <td style="text-align: center; border: none;">125M, 1.3B, 2.7B	</td>
      <td style="text-align: center; border: none;">50k	</td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2021-03 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;">GPT-J </td>
      <td style="text-align: center; border: none;">EleutderAI </td>
      <td style="text-align: center; border: none;">6B 	</td>
      <td style="text-align: center; border: none;">50k	</td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2021-06 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>      
<tr>
      <th style="text-align: center; border: none;">  Codex   </td>
      <td style="text-align: center; border: none;">OpenAI  </td>
      <td style="text-align: center; border: none;">12M, 25M, 42M, 85M, 300M, 679M, 2.5B, 12B </td>
      <td style="text-align: center; border: none;">-	</td>
      <td style="text-align: center; border: none;">4096	 </td>
      <td style="text-align: center; border: none;">2021-07 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeParrot    </td>
      <td style="text-align: center; border: none;">Hugging Face  </td>
      <td style="text-align: center; border: none;">110M, 1.5B </td>
      <td style="text-align: center; border: none;">33k </td>
      <td style="text-align: center; border: none;">1024 </td>
      <td style="text-align: center; border: none;">2021-11 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;"> PolyCoder     </td>
      <td style="text-align: center; border: none;">CMU </td>
      <td style="text-align: center; border: none;">160M, 400M, 2.7B </td>
      <td style="text-align: center; border: none;">50k	</td>
      <td style="text-align: center; border: none;">2048	 </td>
      <td style="text-align: center; border: none;">2022-02 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeGen    </td>
      <td style="text-align: center; border: none;">Salesforce </td>
      <td style="text-align: center; border: none;">350M, 2.7B, 6.1B 16.1B </td>
      <td style="text-align: center; border: none;">51k 	</td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2022-03 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  GPT-NeoX    </td>
      <td style="text-align: center; border: none;">EleutderAI  </td>
      <td style="text-align: center; border: none;">20B	</td>
      <td style="text-align: center; border: none;">50k </td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2022-04  </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  PaLM-Coder    </td>
      <td style="text-align: center; border: none;">Google  </td>
      <td style="text-align: center; border: none;">8B, 62B, 540B  </td>
      <td style="text-align: center; border: none;">256k  </td>
      <td style="text-align: center; border: none;">2048	 </td>
      <td style="text-align: center; border: none;">2022-04 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  InCoder    </td>
      <td style="text-align: center; border: none;">Meta </td>
      <td style="text-align: center; border: none;">1.3B, 6.7B     </td>
      <td style="text-align: center; border: none;">50k	</td>
      <td style="text-align: center; border: none;">2049	 </td>
      <td style="text-align: center; border: none;">2022-04 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  PanGu-Coder     </td>
      <td style="text-align: center; border: none;">Huawei </td>
      <td style="text-align: center; border: none;">317M, 2.6B </td>
      <td style="text-align: center; border: none;">42k </td>
      <td style="text-align: center; border: none;">1024	 </td>
      <td style="text-align: center; border: none;">2022-07 	 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  PyCodeGPT    </td>
      <td style="text-align: center; border: none;">Microsoft </td>
      <td style="text-align: center; border: none;">110M       </td>
      <td style="text-align: center; border: none;">32k </td>
      <td style="text-align: center; border: none;">1024    </td>
      <td style="text-align: center; border: none;">2022-06      </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeGeeX   </td>
      <td style="text-align: center; border: none;">Tsinghua </td>
      <td style="text-align: center; border: none;">13B  	</td>
      <td style="text-align: center; border: none;">52k </td>
      <td style="text-align: center; border: none;">2048	 </td>
      <td style="text-align: center; border: none;">2022-09 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  BLOOM    </td>
      <td style="text-align: center; border: none;">BigScience  </td>
      <td style="text-align: center; border: none;">176B </td>
      <td style="text-align: center; border: none;">251k </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">2022-11 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  ChatGPT    </td>
      <td style="text-align: center; border: none;">OpenAI  </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">16k </td>
      <td style="text-align: center; border: none;">2022-11 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  SantaCoder    </td>
      <td style="text-align: center; border: none;">Hugging Face </td>
      <td style="text-align: center; border: none;">1.1B  </td>
      <td style="text-align: center; border: none;">49k  	</td>
      <td style="text-align: center; border: none;">2048	 </td>
      <td style="text-align: center; border: none;">2022-12 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  LLaMA    </td>
      <td style="text-align: center; border: none;">Meta </td>
      <td style="text-align: center; border: none;">6.7B, 13.0B, 32.5B, 65.2B </td>
      <td style="text-align: center; border: none;">32K </td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2023-02  </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  GPT-4    </td>
      <td style="text-align: center; border: none;">OpenAI  </td>
      <td style="text-align: center; border: none;">-   	</td>
      <td style="text-align: center; border: none;">-	</td>
      <td style="text-align: center; border: none;">32K </td>
      <td style="text-align: center; border: none;">2023-03 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeGen2    </td>
      <td style="text-align: center; border: none;">Salesforce  </td>
      <td style="text-align: center; border: none;">1B, 3.7B, 7B, 16B </td>
      <td style="text-align: center; border: none;">51k </td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2023-05 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  replit-code    </td>
      <td style="text-align: center; border: none;">replit  </td>
      <td style="text-align: center; border: none;">3B   </td>
      <td style="text-align: center; border: none;">33k	</td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2023-05 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;">  StarCoder   </td>
      <td style="text-align: center; border: none;">Hugging Face  </td>
      <td style="text-align: center; border: none;">15.5B   </td>
      <td style="text-align: center; border: none;">49k	</td>
      <td style="text-align: center; border: none;">8192 </td>
      <td style="text-align: center; border: none;">2023-05 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;">  WizardCoder    </td>
      <td style="text-align: center; border: none;">Microsoft </td>
      <td style="text-align: center; border: none;">15B, 34B </td>
      <td style="text-align: center; border: none;">49k  </td>
      <td style="text-align: center; border: none;">8192 </td>
      <td style="text-align: center; border: none;">2023-06 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  phi-1    </td>
      <td style="text-align: center; border: none;">Microsoft </td>
      <td style="text-align: center; border: none;">1.3B    </td>
      <td style="text-align: center; border: none;">51k  	</td>
      <td style="text-align: center; border: none;">2048	 </td>
      <td style="text-align: center; border: none;">2023-06 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeGeeX2    </td>
      <td style="text-align: center; border: none;">Tsinghua  </td>
      <td style="text-align: center; border: none;">6B  </td>
      <td style="text-align: center; border: none;">65k	</td>
      <td style="text-align: center; border: none;">8192 </td>
      <td style="text-align: center; border: none;">2023-07  </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  PanGu-Coder2    </td>
      <td style="text-align: center; border: none;">Huawei  </td>
      <td style="text-align: center; border: none;">15B </td>
      <td style="text-align: center; border: none;">42k </td>
      <td style="text-align: center; border: none;">1024  </td>
      <td style="text-align: center; border: none;">2023-07  </td>
      <td style="text-align: center; border: none;"></td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  Llama 2    </td>
      <td style="text-align: center; border: none;">Meta  </td>
      <td style="text-align: center; border: none;">7B, 13B, 70B   </td>
      <td style="text-align: center; border: none;">32K  </td>
      <td style="text-align: center; border: none;">4096 </td>
      <td style="text-align: center; border: none;">2023-07 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  OctoCoder    </td>
      <td style="text-align: center; border: none;">Hugging Face  </td>
      <td style="text-align: center; border: none;">15.5B </td>
      <td style="text-align: center; border: none;">49k	</td>
      <td style="text-align: center; border: none;">8192 </td>
      <td style="text-align: center; border: none;">2023-08 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  Code Llama    </td>
      <td style="text-align: center; border: none;">Meta  </td>
      <td style="text-align: center; border: none;">7B, 13B, 34B </td>
      <td style="text-align: center; border: none;">32k </td>
      <td style="text-align: center; border: none;">16384 </td>
      <td style="text-align: center; border: none;">2023-08 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeFuse   </td>
      <td style="text-align: center; border: none;">Ant Group </td>
      <td style="text-align: center; border: none;">350M, 13B, 34B  </td>
      <td style="text-align: center; border: none;">101k </td>
      <td style="text-align: center; border: none;">4096	 </td>
      <td style="text-align: center; border: none;">2023-09 	 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  phi-1.5    </td>
      <td style="text-align: center; border: none;">Microsoft  </td>
      <td style="text-align: center; border: none;">1.3B </td>
      <td style="text-align: center; border: none;">51k	</td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2023-09 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;">  CodeShell    </td>
      <td style="text-align: center; border: none;">Peking University </td>
      <td style="text-align: center; border: none;">7B </td>
      <td style="text-align: center; border: none;">70k	</td>
      <td style="text-align: center; border: none;">8192 </td>
      <td style="text-align: center; border: none;">2023-10 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>

<tr>
      <th style="text-align: center; border: none;"> Magicoder    </td>
      <td style="text-align: center; border: none;">UIUC  </td>
      <td style="text-align: center; border: none;">7B </td>
      <td style="text-align: center; border: none;">32k </td>
      <td style="text-align: center; border: none;">16384 </td>
      <td style="text-align: center; border: none;">2023-12 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> AlphaCode 2    </td>
      <td style="text-align: center; border: none;">Google DeepMind  </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">2023-12 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>
<tr>
      <th style="text-align: center; border: none;">  StableCode    </td>
      <td style="text-align: center; border: none;">StabilityAI  </td>
      <td style="text-align: center; border: none;">3B   </td>
      <td style="text-align: center; border: none;">50k	</td>
      <td style="text-align: center; border: none;">16384 </td>
      <td style="text-align: center; border: none;">2024-01  </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> WaveCoder    </td>
      <td style="text-align: center; border: none;">Microsoft  </td>
      <td style="text-align: center; border: none;">6.7B </td>
      <td style="text-align: center; border: none;">32k	</td>
      <td style="text-align: center; border: none;">16384 </td>
      <td style="text-align: center; border: none;">2023-12  </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;">  phi-2    </td>
      <td style="text-align: center; border: none;">Microsoft  </td>
      <td style="text-align: center; border: none;">2.7B  	</td>
      <td style="text-align: center; border: none;">51k </td>
      <td style="text-align: center; border: none;">2048 </td>
      <td style="text-align: center; border: none;">2023-12 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;">  DeepSeek-Coder   </td>
      <td style="text-align: center; border: none;">DeepSeek  </td>
      <td style="text-align: center; border: none;">1.3B, 6.7B, 33B </td>
      <td style="text-align: center; border: none;">32k	</td>
      <td style="text-align: center; border: none;">16384 </td>
      <td style="text-align: center; border: none;">2023-11 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> StarCoder 2    </td>
      <td style="text-align: center; border: none;">Hugging Face </td>
      <td style="text-align: center; border: none;">15B </td>
      <td style="text-align: center; border: none;">49k </td>
      <td style="text-align: center; border: none;">16384 </td>
      <td style="text-align: center; border: none;">2024-02  </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> Claude 3   </td>
      <td style="text-align: center; border: none;">Antdropic </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">- </td>
      <td style="text-align: center; border: none;">200K </td>
      <td style="text-align: center; border: none;">2024-03 </td>
      <td style="text-align: center; border: none;"></td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> CodeGemma   </td>
      <td style="text-align: center; border: none;">Google  </td>
      <td style="text-align: center; border: none;">2B, 7B  </td>
      <td style="text-align: center; border: none;">25.6k </td>
      <td style="text-align: center; border: none;">8192	</td>
      <td style="text-align: center; border: none;">2024-04 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> Code-Qwen   </td>
      <td style="text-align: center; border: none;">Qwen Group </td>
      <td style="text-align: center; border: none;">7B </td>
      <td style="text-align: center; border: none;">92K </td>
      <td style="text-align: center; border: none;">65536 </td>
      <td style="text-align: center; border: none;">2024-04</td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr>
      <th style="text-align: center; border: none;"> Llama3    </td>
      <td style="text-align: center; border: none;">Meta </td>
      <td style="text-align: center; border: none;">8B, 70B </td>
      <td style="text-align: center; border: none;">128K </td>
      <td style="text-align: center; border: none;">8192 </td>
      <td style="text-align: center; border: none;">2024-04 </td>
      <td style="text-align: center; border: none;">✓</td>
    </tr>
<tr style="border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;"> StarCoder2-Instruct    </td>
    <td style="text-align: center; border: none;">Hugging Face </td>
    <td style="text-align: center; border: none;">15.5B </td>
    <td style="text-align: center; border: none;">49K </td>
    <td style="text-align: center; border: none;">16384 </td>
    <td style="text-align: center; border: none;">2024-04 </td>
    <td style="text-align: center; border: none;">✓</td>
</tr>
    </tbody>
</table>
<p style="text-align: center;">Table.2 仅采用解码器架构的大语言模型（LLMs）</p>

##### 方向三：大语言模型的训练
正如前文所述，大型语言模型的卓越性能可归因于它们在大型和多样化数据集上的训练。对于通用LLM而言，积累来自各种来源的大规模自然语言语料库至关重要。这些来源包括网页、对话数据、书籍和新闻、科学数据以及代码，而这些数据通常是从网络上抓取的，必须经过细致而积极的预处理。多个平台和网站提供了大规模、开源且许可宽松的代码语料库，例如GitHub和Stack Overflow。值得注意的是，GitHub存储库的星标或fork数量已成为过滤高质量代码数据集的重要指标。类似地，Stack Overflow上的投票数量可以用来辨别最相关和最优秀的答案。

然而，原始数据集经常包含冗余、嘈杂的数据和个人信息，引起对隐私泄露的担忧，其中可能包括存储库贡献者的姓名和电子邮件地址。因此，必须进行严格的数据清理程序。通常，此过程包括精确匹配去重、基于平均行长度和字母数字字符比例的定义阈值的代码数据过滤、通过关键字搜索删除自动生成的文件，以及删除个人用户数据。具体而言，标准数据预处理工作流程如Fig.1-1.3所示。
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_04.png" alt="Fig.1-1.4"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-1.3 代码生成大语言模型（LLMs）预训练阶段标准数据预处理流程的示意图</p>

文章介绍了 **预训练** 和 **指令调优** 的常用数据集。在初始阶段，代码生成的语言模型通常使用包含人工标注的自然语言描述和相应代码片段的数据集，在一个监督学习框架内从头开始训练。然而，人工标注不仅费力且耗时，而且由此产生的模型的效力也受到可用标注数据的数量和质量的限制。鉴于这些挑战人们已经转向一种替代的训练策略，即在广泛的、未标记的代码语料库上对模型进行预训练。这种方法旨在使模型具有对编程知识的广泛理解，包括标识符、代码结构和底层语义等元素。
<table>
  <thead>
    <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
      <th style="text-align: center; border: none;"><b>Dataset</b></th>
      <th style="text-align: center; border: none;"><b>Size (GB)</b></th>
      <th style="text-align: center; border: none;"><b>Files (M)</b></th>
      <th style="text-align: center; border: none;"><b>#PL</b></th>
      <th style="text-align: center; border: none;"><b>Date</b></th>
      <th style="text-align: center; border: none;"><b>Link</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="text-align: center; border: none;">CodeSearchNet</td>
      <td style="text-align: center; border: none;">20</td>
      <td style="text-align: center; border: none;">6.5</td>
      <td style="text-align: center; border: none;">6</td>
      <td style="text-align: center; border: none;">2022-01</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/code_search_net">https://huggingface.co/datasets/code_search_net</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">Google BigQuery</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">2016-06</td>
      <td style="text-align: center; border: none;"><href url="https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code">github-on-bigquery-analyze-all-the-open-source-code</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">The Pile</td>
      <td style="text-align: center; border: none;">95</td>
      <td style="text-align: center; border: none;">19</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">2022-01</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/EleutherAI/pile">https://huggingface.co/datasets/EleutherAI/pile</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">CodeParrot</td>
      <td style="text-align: center; border: none;">180</td>
      <td style="text-align: center; border: none;">22</td>
      <td style="text-align: center; border: none;">1</td>
      <td style="text-align: center; border: none;">2021-08</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/transformersbook/codeparrot">https://huggingface.co/datasets/transformersbook/codeparrot</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">GitHub Code</td>
      <td style="text-align: center; border: none;">1,024</td>
      <td style="text-align: center; border: none;">115</td>
      <td style="text-align: center; border: none;">32</td>
      <td style="text-align: center; border: none;">2022-02</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/codeparrot/github-code">https://huggingface.co/datasets/codeparrot/github-code</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">ROOTS</td>
      <td style="text-align: center; border: none;">163</td>
      <td style="text-align: center; border: none;">15</td>
      <td style="text-align: center; border: none;">13</td>
      <td style="text-align: center; border: none;">2023-03</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/bigscience-data">https://huggingface.co/bigscience-data</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">The Stack</td>
      <td style="text-align: center; border: none;">3,136</td>
      <td style="text-align: center; border: none;">317</td>
      <td style="text-align: center; border: none;">30</td>
      <td style="text-align: center; border: none;">2022-10</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/bigcode/the-stack">https://huggingface.co/datasets/bigcode/the-stack</href></td>
    </tr>
    <tr style="border-bottom: 1px solid #333;">
      <th style="text-align: center; border: none;">The Stack v2</td>
      <td style="text-align: center; border: none;">32K</td>
      <td style="text-align: center; border: none;">3K</td>
      <td style="text-align: center; border: none;">619</td>
      <td style="text-align: center; border: none;">2024-04</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/bigcode/the-stack-v2">https://huggingface.co/datasets/bigcode/the-stack-v2</href></td>
    </tr>
  </tbody>
</table>
<p style="text-align: center;">Table.3  针对代码生成的LLM的一些常用预训练数据集</p>


在大规模数据集上对大型语言模型进行预训练后，下一个阶段通常涉及增强模型处理和遵循各种指令的能力，这被称为**指令调优**。指令调优通常指的是使用由结构化示例组成的数据集对预训练的大型语言模型进行监督式微调，这些结构化示例被构建为各种自然语言指令。
<table>
  <thead>
    <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
      <th style="text-align: center; border: none;"><b>Dataset</b></th>
      <th style="text-align: center; border: none;"><b>Size</b></th>
      <th style="text-align: center; border: none;"><b>#PL</b></th>
      <th style="text-align: center; border: none;"><b>Date</b></th>
      <th style="text-align: center; border: none;"><b>Link</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="text-align: center; border: none;">CodeAlpaca-20K </td>
      <td style="text-align: center; border: none;">20k</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">2023-03</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k">https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">CommitPackFT</td>
      <td style="text-align: center; border: none;">2GB</td>
      <td style="text-align: center; border: none;">277</td>
      <td style="text-align: center; border: none;">2023-08</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/bigcode/commitpackft">https://huggingface.co/datasets/bigcode/commitpackft</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">Evol-Instruct-Code-80k</td>
      <td style="text-align: center; border: none;">80k</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">2023-07</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1">https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">evol-codealpaca-v1 </td>
      <td style="text-align: center; border: none;">110K</td>
      <td style="text-align: center; border: none;">-</td>
      <td style="text-align: center; border: none;">2023-07</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1">https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1</href></td>
    </tr>
    <tr>
      <th style="text-align: center; border: none;">Magicoder-OSS-Instruct-75k</td>
      <td style="text-align: center; border: none;">75k</td>
      <td style="text-align: center; border: none;">
          Python, Shell, TypeScript, C++, Rust, PHP, Java, Swift, C#
      </td>
      <td style="text-align: center; border: none;">2023-12</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K">https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K</href></td>
    </tr>
    <tr style="border-bottom: 1px solid #333;">
      <th style="text-align: center; border: none;">Self-OSS-Instruct-SC2-Exec-Filter-50k</td>
      <td style="text-align: center; border: none;">50k</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2024-04</td>
      <td style="text-align: center; border: none;"><href url="https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k">https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k</href></td>
    </tr>
  </tbody>
</table>
<p style="text-align: center;">Table.4 用于指令调整LLM以进行代码生成的几个代表性数据集 </p>

此外，文章还介绍了**检索增强生成(RAG)** 和**基于人类反馈的强化学习(RLHF)** 等方法，在第三部分的大模型调研中将详细说明。
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_03.png" alt="Fig.1-1.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-1.4 代码大语言模型（Code LLMs）及其关联数据库通用训练、推理与评估流程的示意图。该训练流程主要分为四个独立阶段：阶段①和②属于预训练阶段，而阶段③和④代表训练后阶段。需特别说明的是，阶段②和④为可选流程。</p>

#### 相关论文二：[A Systematic Survey on Large Language Models for Algorithm Design](https://arxiv.org/abs/2410.14716)
文章围绕大语言模型在算法设计（LLM4AD）的应用展开系统调研。先阐述算法设计重要性及 LLMs 给该领域带来的变革，强调对其系统综述的必要性。通过明确研究范围，经多阶段收集 180 余篇相关论文。
文章提出涵盖 LLM 角色、搜索方法、提示方法和应用领域的多维度分类法。在 LLM 角色上，分为优化器、预测器、提取器和设计者，分别阐述其任务、优势与局限。搜索方法包含采样、单点搜索、基于种群的搜索和不确定性引导搜索等多种方式。提示方法介绍零样本、少样本等策略及其在算法设计中的应用。
应用领域方面，详细探讨 LLMs 在优化、机器学习、科学发现和工业等领域的具体应用，展示其在不同场景下的成果与潜力。同时，分析 LLMs 在算法设计面临的如可扩展性、可解释性等挑战，并提出未来研究方向，包括开发领域特定 LLMs、探索多模态 LLMs 等。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig1.1.png" alt="Fig1.1"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-2.1 文章的主要研究方向</p>

##### 1.大语言模型在算法设计领域的角色（定位）

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig1.2.png" alt="Fig1.2"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-2.2 大语言模型作为优化器</p>

**LLMs 作为优化器（LLMaO）**：任务是在算法框架中作为黑箱优化器，生成并优化解决方案。它利用 LLMs 理解和生成复杂模式与解决方案的能力以及良好的灵活性，应用于传统优化任务、自动提示优化等领域。例如 Yang 等利用 LLMs 的上下文学习能力为特定问题生成新解决方案，并迭代优化；Liu 等将 LLMs 作为进化算子解决多目标问题。例如在组合优化问题中，针对旅行商问题（TSP），它要生成不同的城市遍历路径方案，并逐步改进路径，使总路程最短。在自动提示优化场景下，它会生成不同的提示组合，通过不断调整提示内容，提升大语言模型在特定任务上的输出质量。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig1.3.png" alt="Fig1.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-2.3 大语言模型作为预测器</p>

**LLMs 作为预测器（LLMaP）**：主要任务是作为替代模型，在分类或回归任务中预测解决方案的结果或响应。它能够处理和生成类似人类的响应，理解和解释数据中的复杂模式，并且预训练的 LLMs 可显著减少计算负荷和时间。LLMs 在大量包含解决方案及其对应结果的数据上进行训练，学习到数据中的复杂模式和关系。当输入新的解决方案相关信息时，模型将其与训练数据中的模式进行匹配和关联，通过内部的神经网络结构进行复杂计算，输出对结果的预测值。如 Jawahar 等用 LLMs 预测深度神经网络架构的性能；Hao 等利用 LLMs 作为进化算法中的替代模型进行回归和分类

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig1.4.png" alt="Fig1.4"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-2.3 大语言模型作为提取器</p>

**LLMs 作为提取器（LLMaE）**：负责从目标问题和/或算法中挖掘和提取嵌入特征或特定知识，以增强基于算法的问题解决能力。在分析复杂算法时，它要找出算法中关键的操作步骤、数据处理流程等知识；在处理自然语言描述的编程问题时，提取出问题中的关键概念、约束条件等特征。当面对目标问题或算法描述时，模型通过对文本的解析，识别出关键的词汇、短语以及它们之间的语法和语义关系，从而提取出有价值的特征和知识。例如在研究一个新的排序算法时，它提取出算法中核心的比较、交换操作以及这些操作所依赖的数据结构特征；Kristiadi 等将 LLMs 用作预训练特征提取器，增强标准贝叶斯优化替代模型；Wu 等利用 LLMs 提取高维算法表示，确定最适合特定问题的算法

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig1.5.png" alt="Fig1.5"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.1-2.3 大语言模型作为设计者</p>

**LLMs 作为设计者（LLMaD）**：直接创建算法或特定组件，如生成启发式算法、编写代码片段或制定函数等，能够显著加速算法设计过程，减少人力投入，并为算法开发带来创造性和优化。它可以根据给定的问题需求和目标，生成全新的算法逻辑；也能针对已有算法，生成特定的功能组件，如生成排序算法中的比较函数、搜索算法中的启发式函数等。通过对大量代码和算法数据的学习，LLMs掌握了丰富的编程模式、算法结构和设计思路。当接收到设计任务需求时，模型基于对需求的理解，从学习到的知识中选取合适的模式和结构进行组合与创新，生成符合要求的算法或组件代码。例如， Eureka 利用 LLMs 的代码编写和上下文学习能力，进化和优化强化学习的奖励函数；ADAS 提出自动设计智能系统，通过元代理生成强大的智能系统设计.
##### 2.搜索方法
在算法设计中，将大语言模型（LLMs）融入搜索框架可提升其效用。本部分对现有研究按搜索方法分类，介绍进展并探讨局限。
 - **采样**：最直接的搜索方式是让LLM重复采样新设计，选取最佳样本作为最终设计，但简单采样成本较高，包括束搜索和蒙特卡罗树搜索（MCTS）*等。
 - **基于单点的搜索**：该方法通过利用邻域结构或特定搜索方向迭代优化解决方案，但在搜索过程中难以保持多样性和鲁棒性，具体方法有爬山搜索、邻域搜索、基于梯度的搜索以及强化学习等。
 - **基于种群的搜索**：基于种群的进化搜索因其在复杂搜索空间的有效性和鲁棒性，成为LLM4AD研究的主要工具，多数研究使用简单遗传算法和贪婪种群管理，部分探索了先进的种群管理方法，具体方法有单目标进化搜索和多目标进化搜索。
 - **不确定性引导的搜索**：该方法将贝叶斯最优实验设计（BOED）或贝叶斯优化（BO）与LLMs结合，从基于初始参数信念的先验开始，通过不确定性驱动策略迭代优化信念，在多种应用中展现出有效性，如提取环境特征、优化多轮决策推理、LLM解码等。

##### 3.提示策略
提示策略对有效利用LLMs至关重要，尤其是在算法设计这类需要推理和反思的任务中。现有LLM4AD研究中，超80%使用预训练模型且多数选择GPT模型，涉及多种提示工程方法。
 - **零样本**：零样本提示使模型无需针对特定任务训练就能理解和执行任务，在算法设计中可直接请求LLM提供解决方案，但可能无法满足复杂算法任务的细致需求。
 - **少样本**：少样本提示通过提供少量示例帮助模型理解任务背景，在算法设计中，这些示例包括算法、解决方案、提示和代码等，可手动设计或由LLM生成，且通常会进行排序以提升性能。
 - **思维链**：思维链提示鼓励模型阐述得出最终答案的中间步骤或推理路径，在算法设计中有助于理解设计过程、避免异常结果，如引导LLM推理现有启发式方法、评估步长合理性等。
 - **自一致性**：自一致性通过让模型对同一提示生成多个答案并综合，提高准确性和可靠性，在算法设计中表现为多次请求模型解决问题并比较解决方案，以确定更高效或稳健的算法。
 - **反思**：反思是指让模型评估自身的响应或解决方案，在算法设计中，用于分析算法的效率、潜在缺陷和改进方向，如在提示优化、启发式设计和强化学习奖励函数设计中都有应用。

##### 4.应用领域
LLMs在多个领域的算法设计中有着广泛应用，涵盖优化、机器学习、科学发现和工业等方面。
 - **优化**：LLMs在优化领域的应用广泛，包括组合优化、连续优化、贝叶斯优化、提示优化和优化建模等，不同方法利用LLMs的不同角色和提示策略解决各类优化问题，还在算法选择和代码生成等方面有应用。
    - 具体包括组合优化、连续优化、贝叶斯优化、提示优化、优化建模等应用。
 - **机器学习**：LLMs在机器学习领域的应用涉及任务规划、强化学习、神经架构搜索、图学习、数据集标注等多个方面，为算法设计带来新的思路和方法。
    - 具体包括任务规划、强化学习、神经架构搜索、图学习、数据集标注等应用。
 - **科学发现**：LLMs在科学发现领域的应用与算法设计紧密相关，涉及一般科学发现、化学、生物学、物理学和力学等多个学科，通过搜索方程、设计分子、预测蛋白质相互作用等方式推动科学研究。
    - 具体包括化学、生物学与物理学等学科在内的各种设计、发现类问题。
 - **工业**：LLMs在工业领域的算法设计中具有变革性影响，应用于构建6G网络系统、电子设计自动化、云服务故障分析、多种工业设计和行程规划等方面，但也面临一些挑战。
    - 具体包括网络系统构造、电子设计自动化、云服务故障分析、工业设计与行程规划等内容。 

##### 5.挑战​
- **性能与成本**：简单采样搜索成本高，复杂搜索方法在实际应用中计算开销大，如基于种群的进化搜索虽有效，但计算资源消耗多。此外，训练和使用大规模预训练模型成本高昂，限制了模型的广泛应用和进一步优化。​
- **可解释性**：LLMs 作为黑盒模型，其决策过程和生成结果的原理难以理解，如在作为优化器和预测器时，难以解释解决方案和预测结果是如何得出的，这在对解释性要求较高的领域，如医疗、金融等，限制了模型的应用。​
- **可靠性与准确性**：模型生成的结果可能存在错误或不一致性，如在算法设计中生成的代码可能无法运行或存在漏洞，在科学发现领域提出的假设可能不准确，影响了模型在关键任务中的应用。​
- **领域特定知识**：在处理专业领域问题时，LLMs 缺乏足够的领域特定知识，难以满足复杂专业任务的需求，如在工业设计、医学研究等领域，需要结合专业知识进行算法设计和问题解决。​
- **数据隐私与安全**：在利用大量数据进行训练和应用过程中，存在数据隐私泄露和安全风险，如在优化建模和数据集标注等应用中，如何保护数据隐私和确保数据安全是亟待解决的问题。​
##### 6.未来方向​
- **可解释性研究**：开展 LLMs 的可解释性研究，开发可视化工具和解释方法，帮助用户理解模型的决策过程和生成结果的依据，增强用户对模型的信任，推动模型在高风险领域的应用。​通过改进模型训练方法、引入验证和纠错机制等，提高模型生成结果的可靠性和准确性，如采用多轮验证、自动测试等方式，确保算法设计和代码生成的质量。​
- **领域知识融合**：将领域特定知识融入 LLMs，通过知识图谱、领域数据增强等方式，提升模型在专业领域的表现，满足不同行业的实际需求，如开发针对医疗、金融等领域的专业模型。​
- **隐私保护技术**：研究数据隐私保护技术，如联邦学习、差分隐私等，在保证数据安全的前提下，充分利用数据进行模型训练和应用，推动 LLM4AD 在敏感数据领域的发展。​
- **多模态融合**：进一步探索多模态信息的融合，如结合文本、图像、音频等数据，丰富模型的输入信息，提升模型的泛化能力和应用范围，如在工业设计中结合图像和文本信息进行产品设计。​

****
## 主题二：增强大语言模型推理能力的提示策略


近年来，大语言模型（Large Language Models, LLMs）在自然语言处理（NLP）领域取得了显著进展，展现出强大的文本生成、问答和语义理解能力。然而，尽管这些模型在诸多任务中表现优异，其推理能力——尤其是复杂逻辑推理、多步问题解决和因果推断——仍然存在明显局限性。例如，模型可能在数学推理、常识推理或需要长期依赖的任务中表现不佳，甚至生成看似合理但逻辑错误的答案。这一局限性部分源于模型训练数据的静态性以及自回归生成方式的局部性，同时也与提示（prompting）策略的设计密切相关。
提示策略作为用户与模型交互的核心媒介，直接影响模型的输出质量。早期的提示方法（如零样本或小样本提示）虽然简单有效，但在复杂任务中往往无法充分激发模型的潜力。因此，研究者开始探索更高效的提示策略，旨在通过结构化指令、思维链（Chain-of-Thought, CoT）引导或外部工具协同等方式，显式增强模型的推理能力。这一研究方向不仅对提升模型的实际应用价值具有重要意义，也为理解模型的内在机制提供了新的视角。
该领域的挑战包括提示策略的泛化性、对领域知识的依赖，以及计算效率的权衡。未来研究可能进一步探索神经符号结合、跨任务迁移学习，以及基于认知科学的提示设计理论。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_06.png" alt="Fig.1-1.6"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.2-0.1 基于提示工程的大语言模型（LLM）自优化代码生成流程示意图。该过程通过整合执行结果实现迭代式自我优化，并包含可选的自我反思机制以提升生成代码质量。</p>

#### 相关论文一:[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
语言模型规模扩大带来诸多好处，但在算术、常识和符号推理等挑战性任务上表现仍不佳。本文旨在探索通过简单方法解锁大语言模型的推理能力。从发表时间来看，这篇文章是思维链相关研究的开山之作。
**什么是思维链？**
**思维链（Chain of Thought, CoT）** 是一种通过逐步推理来解决问题的方法，尤其在人工智能（如大语言模型）中广泛应用。它通过将复杂问题拆解为多个中间步骤，模拟人类“一步一步思考”的过程，从而提高逻辑推理和问题解决的准确性。

**核心特点**，思维链的核心特点是包括**显式步骤**，**模仿人类推理**的**提升模型性能**技术，该方法将思考过程分解为可解释的中间步骤（如“首先…然后…最后…”），而非直接输出最终答案，推理过程类似人类解题时的逐步推导，例如数学题中先列已知条件，再分步计算。对于大语言模型（如GPT、PaLM），思维链方法能显著改善需要逻辑、数学或多步推理的任务。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.1.png" alt="Fig2.1"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.2-2.1 思维链方法与传统提示词对比</p>

**示例对比**
###### 传统直接回答
- **问题**：小明有5个苹果，吃了2个，又买了8个，现在有多少个？  
- **回答**：3个（缺乏过程，可能因跳跃而出错）。
###### 思维链回答：
1. 小明最初有5个苹果。
1. 吃掉2个后剩余：5 - 2 = **3个**。
1. 又买了8个，现在有：3 + 8 = **11个**。  
1. 最终答案：11个（步骤清晰，可验证）。

**思维链为什么有效？**
首先，人类解决复杂问题时，会自然地将问题拆解为子步骤（如数学题的中间运算）。思维链强制模型显式生成这些步骤，与人类认知模式对齐，减少“跳跃式错误”。直接生成最终答案时，模型可能从海量可能性中随机采样（易出错）。分步推理将问题分解为更确定的子任务（如先减后加），缩小搜索空间。
此外，大语言模型的参数中存储了大量隐式逻辑规则（如算术、因果推理）。思维链通过逐步提示显式激活这些知识，而非依赖端到端的模糊映射。语言模型预训练时接触过大量人类分步推理文本（如教科书、解题过程），思维链利用了这种数据分布的偏好。
最后，多步任务（如代数方程）天然需要中间状态。思维链的结构与任务结构一致，避免“一步到位”的假设。显式步骤让用户能定位错误的位置和类型，而直接答案难以诊断。

**思维链的应用场景**
包括**数学计算**、**逻辑推理**、**代码生成**等，尤其在追踪中间变量时，思维链方法对于推理正确性的提示比较明显。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.2.png" alt="Fig2.1"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.2-2.2 思维链方法示例</p>

思维链的核心是“让思考过程可见”，这种结构化推理显著提升了AI和人类在复杂任务中的表现。大模型在思维链提示下，域内和域外测试解决率大幅提升，如 PaLM 540B 在最后一个字母拼接任务（4 词，域外）中，思维链提示解决率达 94.8%，远超标准提示的 0.2% 。在 GSM8K 基准测试中，思维链提示准确率达 56.9%，远超标准提示的 17.9% 。

#### 相关论文二:[Towards Better Chain-of-Thought Prompting Strategies: A Survey](https://arxiv.org/abs/2310.04959)
大语言模型（LLM）结合提示策略在自然语言处理任务中表现出色，但普通提示策略在多步任务上仍存在局限。思维链（CoT）提示作为一种新兴策略，通过逐步推理提升了 LLM 在多步推理任务上的性能，引起了广泛研究。本文系统分析了影响 CoT 提示效果的四个关键因素：任务类型（如封闭域推理、开放域推理和代码生成等任务对 CoT 提示的响应不同）、提示设计（示范和文本指令的设计影响提示效果）、扩展策略（集成、子问题划分、外部辅助和合理化策略可增强提示性能）和模型（模型大小和训练语料库影响 CoT 提示效果）。此外，文章还探讨了 CoT 提示面临的挑战，包括忠实性、通用性、自合理化、推理分析和理论分析等方面，并提出了未来的研究方向，为相关研究提供了全面参考。
下图为该文章总结的四威廉领域的相关研究，**本文将主要讨论文章中提到的影响思维链提示效果的因素**。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.3.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.2-2.3 根据任务区分不同的思维链</p>

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.4.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.2-2.4 根据提示方法/补充策略对思维链进行分类</p>

##### 1.任务类型

<table>
  <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;">任务类型</th>
    <th style="text-align: center; border: none;">特点</th>
    <th style="text-align: center; border: none;">CoT 提示效果</th>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">封闭域推理和问答</th>
    <td style="text-align: center; border: none;">问题包含所有必要条件和背景知识
    </td>
    <td style="text-align: center; border: none;">能提供推理模式，在数学推理、符号推理和表格问答等任务中表现出色</td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">开放域推理和问答</th>
    <td style="text-align: center; border: none;">基于大规模非结构化知识库回答问题，依赖 LLM 知识质量</td>
    <td style="text-align: center; border: none;">效果因任务而异，不当使用可能降低性能</td>
  </tr>
  <tr style="border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;">代码生成</th>
    <td style="text-align: center; border: none;">根据输入指令生成代码</td>
    <td style="text-align: center; border: none;">与 CoT 的逐步推理链相契合</td>
  </tr>
</table>
<p style="text-align: center;">Table.3 CoT按照类型是否有效</p>

##### 2.提示设计
**示范**：示范是（问题、推理依据、答案）三元组。从问题角度，复杂度高、相关性强且多样的示范问题有助于提升提示性能；从推理依据角度，结构完整、有效的推理依据能促进提示效果，但有效提示不一定要完全正确的推理依据；从整体角度，示范的数量和顺序会影响模型性能，一般 2 个示范效果较好，顺序影响因模型、任务和数据集而异。
**文本指令**：明确的文本指令如 “Let’s think step by step” 能引导 LLM 进行逐步推理，零样本时效果显著，与少样本 CoT 结合可进一步提升性能。

##### 3.扩展策略
**集成**：结合多样的学习器提升模型性能，分为提示集成和预测集成，预测集成性能提升更明显，但计算成本高，选择策略取决于示范数量和计算资源。
**子问题划分**：将复杂问题分解为简单子问题，便于模型解决更难的问题，且能减少无关信息干扰，方便部署不同模块和引入外部辅助。
**外部辅助**：引入外部知识、工具或代码解释器，可扩展 LLM 能力，如在常识问答中注入知识，在复杂计算或搜索任务中借助工具。
**合理化**：纠正 LLM 预测推理依据中的错误，手动合理化成本高，也可使用提示引导模型重新思考，但难以处理导致正确答案的不完美推理依据。

#### 相关论文三:[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
ReAct是一种将推理和行动相结合的方法，旨在解决各种语言推理和决策任务。该方法通过让大语言模型生成推理痕迹和任务特定行动，实现两者的协同作用。在 HotPotQA、FEVER、ALFWorld 和 WebShop 等任务上的实验表明，ReAct 优于仅进行推理或行动的基线方法，能有效减少幻觉和错误传播，提高模型的可解释性和可信度，且在少样本学习设置下表现出色。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.5.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.6.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig2-2.4&2.5 ReAct思维链的基本格式</p>

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig2.7.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">

<p style="text-align: center;">Fig2-2.7 Reason和Act共同协同增加推理可靠性</p>

#### 相关论文四:[Structured Chain-of-Thought Prompting for Code Generation](https://dl.acm.org/doi/10.1145/3690635)
文章提出结构化思维链（SCoT）及 SCoT 提示技术用于代码生成。SCoT 利用顺序、分支和循环结构构建中间推理步骤，SCoT 提示让大语言模型（LLMs）先生成 SCoT 再输出代码。在 HumanEval、MBPP 和 MBCPP 基准测试中，SCoT 提示比思维链（CoT）提示的Pass@1 最高提升 13.79%，且生成的程序更受开发者青睐，对示例更具鲁棒性。

##### 基本思想——SCoT 和 SCoT 提示技术
SCoT：由输入输出（IO）结构和基于顺序、分支、循环三种编程结构的粗略问题解决过程组成。通过明确生成编程结构，解锁 LLMs 的编程能力，且作为自然语言和代码间的桥梁，简洁高效。例如，在处理读取文件需求时，可通过编程结构设计 “若文件存在则读取，否则报错” 的解决思路。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig_s.1.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig2-2.8 SCoT模仿了程序设计中需要的循环、分支结构</p>

SCoT 提示：基于 SCoT 提出的代码生成提示技术，其提示包含自然语言指令、<需求，SCoT, 代码> 示例以及测试要求。通过这种提示，让 LLMs 先生成 SCoT，再生成最终代码。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Fig_s.2.png" alt="Fig2.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig2-2.9 SCoT在Python和C++上的示例</p>

##### 研究设计
**研究问题（RQs）**：共提出 4 个问题，包括 SCoT 提示与基线相比的准确性、开发者对 SCoT 提示生成程序的偏好、SCoT 提示对示例的鲁棒性以及不同编程结构在 SCoT 提示中的贡献。
数据集：选用三个代表性代码生成基准数据集，具体信息如下：

<table>
  <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;">数据集</th>
    <th style="text-align: center; border: none;">语言</th>
    <th style="text-align: center; border: none;">训练集数量</th>
    <th style="text-align: center; border: none;">测试集数量</th>
    <th style="text-align: center; border: none;">平均每个样本测试用例数</th>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">HumanEval</th>
    <td style="text-align: center; border: none;">Python
    </td>
    <td style="text-align: center; border: none;">-</td>
    <td style="text-align: center; border: none;">164</td>
    <td style="text-align: center; border: none;">7.7</td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">MBPP</th>
    <td style="text-align: center; border: none;">Python</td>
    <td style="text-align: center; border: none;">474</td>
    <td style="text-align: center; border: none;">500</td>
    <td style="text-align: center; border: none;">3</td>
  </tr>
  <tr style="border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;">MBCPP</th>
    <td style="text-align: center; border: none;">C++</td>
    <td style="text-align: center; border: none;">413</td>
    <td style="text-align: center; border: none;">435</td>
    <td style="text-align: center; border: none;">3</td>
  </tr>
</table>

<p style="text-align: center;">Table.4 该思维链实验中用到的数据集</p>

**评估指标**：采用 Pass@k 衡量生成程序的正确性，计算生成程序通过所有测试用例的需求占总需求的百分比，k 设为 1、3、5 。同时使用无偏 Pass@k 减少方差。
**对比基线（标准）**：选择零样本提示、少样本提示和 CoT 提示作为基线，确保与 SCoT 提示的示例数量和种子相同，以保证比较的公平性。选取 gpt-4-turbo、gpt-3.5-turbo 和 DeepSeek Coder-Instruct 系列（1.3B、6.7B、33B）共 5 种流行的 LLMs 进行实验。
**采样设置**：使用核采样从 LLMs 中解码程序，所有方法每个需求生成 20 个程序，提示采用固定的 3 个示例，温度设为 0.8，top-p 设为 0.95。
**实验结果与分析**：
- 准确性（RQ1）：在三个基准测试和五种 LLMs 上，SCoT 提示均显著优于基线。在 Pass@1 指标上，相比 CoT 提示，在 HumanEval 中最高提升 13.79%，MBPP 中最高提升 12.31%，MBCPP 中最高提升 13.59%。
- 开发者偏好（RQ2）：通过 10 位开发者对程序的人工评估，发现 SCoT 提示生成的程序在正确性上比 CoT 提示高出 15.27%，在代码bad smell方面减少 36.08%，更受开发者青睐。
- 鲁棒性（RQ3）：SCoT 提示在示例种子、写作风格、示例顺序和示例数量方面表现出更强的鲁棒性，方差更低，性能更优。
- 结构贡献（RQ4）：通过消融研究发现，基本结构（顺序、分支、循环）有助于设计可行的解决过程，去除后 Pass@1 最高下降 8.2%；IO 结构有助于理解需求，去除后 Pass@1 最高下降 2.37%
##### 关键问题：
**SCoT 提示技术与其他提示技术相比，优势主要体现在哪些方面？**
**答案：**SCoT 提示技术优势明显。在准确性上，相比 CoT 提示，在 HumanEval、MBPP 和 MBCPP 基准测试中，Pass@1 最高分别提升 13.79%、12.31% 和 13.59%。在生成程序质量方面，经开发者评估，SCoT 提示生成的程序在正确性上比 CoT 提示高出 15.27%，代码坏味道减少 36.08%。此外，SCoT 提示对示例种子、写作风格、示例顺序和数量更具鲁棒性。
**SCoT 的结构组成对代码生成有怎样的作用？**
**答案：**SCoT 由 IO 结构和基于顺序、分支、循环的问题解决过程组成。IO 结构明确了代码的输入输出，有助于理解需求，删除该结构会使 SCoT 提示的 Pass@1 最高下降 2.37%。顺序、分支和循环这三种基本编程结构，能帮助 LLMs 清晰地设计解决过程，删除后 Pass@1 最高下降 8.2%。它们使 SCoT 更清晰、更接近代码，有利于后续代码实现。
****
## 主题三：大语言模型应用
#### 论文阅读一：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v1)

#### 论文阅读二：[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

DeepSeek-V3/R1模型架构如下
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/ds_struct.png" alt="Fig.3-0.1"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-0.1 DeepSeek-V3/R1模型架构</p>

根据Deepseek Github文档，可以得到Deepseek中MLA的定义
<pre><code class="language-python">
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim #隐藏层维度
        self.n_heads = args.n_heads 
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank #q的低秩压缩的维度
        self.kv_lora_rank = args.kv_lora_rank #kv的低秩压缩的维度
        self.qk_nope_head_dim = args.qk_nope_head_dim #qk不带旋转位置编码的头的维度
        self.qk_rope_head_dim = args.qk_rope_head_dim #qk旋转位置编码的头的维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim #v的多头注意力中头的维度
        
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        #q的down-projection矩阵
        
        self.q_norm = nn.RMSNorm(self.q_lora_rank)
        
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        #q的up-projection矩阵
        
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        # wkv_a为K和V的down-projection矩阵
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)
        
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        # wkv_b为K和V的up-projection矩阵
        
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim) #output权重矩阵
        self.softmax_scale = self.qk_head_dim ** -0.5#计算1/sqrt(d_k)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
        
    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.size()
        start_pos = 1
        end_pos = start_pos + seqlen
        # ---- 计算q--------
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) #分离nope,rope
        q_pe = apply_rotary_emb(q_pe, freqs_cis) #执行RoPE计算
        
        # ----计算KV----------
        kv = self.wkv_a(x)
        #KV-Cache大小为wkv_a outputdim(self.kv_lora_rank + self.qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) #分离KV和K位置编码
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) #执行RoPE计算
        
        # -----处理KV u-pprojection矩阵
        wkv_b = self.wkv_b.weight 
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        
        # q中不需要位置编码的先和K的不需要位置编码的权重相乘
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)#保存KV Cache
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2) #保存K的位置编码Cache(pe cache)
        
        # 计算QK^T/sqrt(d_k)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # 计算V
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        x = self.wo(x.flatten(2)) #wo权重, 从n_head * v_head_dim -> dim
        return x
</code></pre>

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/ds_cal1.png" alt="Fig.3-0.1"
style="display: block; margin: 0 auto;">
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/ds_cal2.png" alt="Fig.3-0.1"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-0.2 上图展示了计算Deepseek模型吞吐和性能的计算流程，单个Token的KVCache可以从forward函数中的kv=self.wkv_a(x)得知</p>

##### Deepsee模型的使用渠道
<table>
  <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
    <td style="text-align: center; border: none;"></td>
    <th style="text-align: center; border: none;">平台</th>
    <th style="text-align: center; border: none;">网址链接</th>
  </tr>
  <tr>
    <th rowspan="4" style="text-align: center; border: none;">官方渠道</th>
    <td style="text-align: center; border: none;">官方网页版
    </td>
    <td style="text-align: center; border: none;"><a href="https://chat.deepseek.com" target="_blank">https://chat.deepseek.com</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">官方API版
    </td>
    <td style="text-align: center; border: none;"><a href="https://platform.deepseek.com" target="_blank">https://platform.deepseek.com</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">IOS
    </td>
    <td style="text-align: center; border: none;"><a href="https://apps.apple.com/cn/app/deepseek/id6737597349" target="_blank">https://apps.apple.com/cn/app/deepseek/id6737597349</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">安卓软件
    </td>
    <td style="text-align: center; border: none;"><a href="https://app.mi.com/details?id=com.deepseek.chat" target="_blank">https://app.mi.com/details?id=com.deepseek.chat</a></td>
  </tr>
  <tr>
    <th rowspan="7" style="text-align: center; border: none;">第三方渠道</th>
    <td style="text-align: center; border: none;">国家超算平台（网页）</td>
    <td style="text-align: center; border: none;"><a href="https://chat.scnet.cn/" target="_blank">https://chat.scnet.cn/</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">硅基流动（网页+API）
    </td>
    <td style="text-align: center; border: none;"><a href="https://cloud.siliconflow.cn/i/9VzvgYQL" target="_blank">https://cloud.siliconflow.cn/i/9VzvgYQL</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">阿里百炼（API）
    </td>
    <td style="text-align: center; border: none;"><a href="https://account.aliyun.com/" target="_blank">https://account.aliyun.com/</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">火山引擎（API）
    </td>
    <td style="text-align: center; border: none;"><a href="https://www.volcengine.com/product/ark" target="_blank">https://www.volcengine.com/product/ark</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">AskManyAI（网页）
    </td>
    <td style="text-align: center; border: none;">-</td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">纳米AI搜索（网页）
    </td>
    <td style="text-align: center; border: none;"><a href="https://www.n.cn/" target="_blank">https://www.n.cn/</a></td>
  </tr>
  <tr>
  <td style="text-align: center; border: none;">秘塔AI搜索（网页）
    </td>
    <td style="text-align: center; border: none;"><a href="https://metaso.cn/" target="_blank">https://metaso.cn/</a></td>
  </tr>
</table>

<p style="text-align: center;">Table.5 两种幻觉的示例</p>

Deepseek代码生成方向的场景应用：
1. **生成代码**：用 Python 写一个 [XXX] 脚本，要求实现 [XXX]功能并添加异常处理模块，确保代码正确性，能够处理运行时的错误并给出提示。
2. **API对接**：编写调用 [XXX] 接口的示例代码，包含身份验证和错误重试机制，确保接口调用稳定可靠，适应网络波动等异常情况。
3. **DEBUG助手**：提供解释下面这段代码报错的原因（附错误日志），并给出两种修复方案，帮助开发者快速定位问题并优化代码。
4. **代码审查**：检查以下代码的5个潜在问题，按安全性、性能、可读性分类说明，帮助开发者提升代码质量，避免安全隐患和性能瓶颈。
5. **算法优化**：将 O(n²) 时间复杂度算法优化至 O(n log n)，保留详细注释，解释优化思路和关键步骤，提升代码效率并便于后续维护。
6. **多线程实现**：使用 Python 的 threading 或 asyncio 模块实现[XXX]功能的多线程版本，确保线程安全，优化性能，避免数据竞争和死锁问题。
7. **单元测试编写**：为 [XXX] 模块编写单元测试代码，覆盖核心功能和边界情况，使用断言验证结果，确保代码修改后功能正常，提升代码的可维护性。
8. **数据库迁移脚本**：编写一个数据库迁移脚本，将 [原数据库] 数据迁移到 [目标数据库]，支持数据清洗和格式转换，确保迁移过程无数据丢失。

#### 问题一：大语言模型代码生成的测试基准(benchmarks)
为严格评估大语言模型（LLM）的代码生成能力，近年来研究界陆续提出了多种高质量基准测试。基于的开创性工作，HumanEval数据集衍生出众多变体及新增基准，旨在更全面地评估LLM的代码生成能力。我们根据应用场景将这些基准测试大致划分为六大类：通用编程、竞赛编程、数据科学、多语言支持、逻辑推理以及仓库级代码生成。
在自然语言处理领域，构建可靠且鲁棒的生成内容自动评估指标一直是长期存在的挑战。早期研究大多直接采用基于文本匹配的指标（如精确匹配、BLEU、ROUGE和METEOR等）来评估代码生成质量，这些指标虽能快速经济地评估生成代码，但往往难以准确捕捉代码的语法正确性、功能完备性和语义特征。
当前主流评估转向基于执行的指标，包括：pass@k，其衡量模型生成的k个代码样本中至少一个通过所有单元测试的概率，其无偏估计公式为：
$$pass@k:=E_{task}[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}] $$

其中$n$为候选代码总数，$k$为抽样数量$（n\ge k）$，$c$为$k$个样本中的正确解数量。其他指标还有$n@k$、测试用例平均通过率、执行准确率、$pass@t$等
<table>
  <thead>
    <tr style="border-top: 1px solid #333;">
      <th style="text-align: center; border: none;">Scenario</th>
      <th style="text-align: center; border: none;">Benchmark</th>
      <th style="text-align: center; border: none;">Size</th>
      <th style="text-align: center; border: none;">#PL</th>
      <th style="text-align: center; border: none;">Date</th>
      <th style="text-align: center; border: none;">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-top: 1px solid #333;">
      <td rowspan="15" style="text-align: center; border: none;">General</td>
      <td style="text-align: center; border: none;">HumanEval</td>
      <td style="text-align: center; border: none;">164</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2021-07</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/openai_humaneval">https://huggingface.co/datasets/openai_humaneval</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">HumanEval+</td>
      <td style="text-align: center; border: none;">164</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2023-05</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/evalplus/humanevalplus">https://huggingface.co/datasets/evalplus/humanevalplus</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">HumanEvalPack</td>
      <td style="text-align: center; border: none;">164</td>
      <td style="text-align: center; border: none;">6</td>
      <td style="text-align: center; border: none;">2023-08</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/bigcode/humanevalpack">https://huggingface.co/datasets/bigcode/humanevalpack</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">MBPP</td>
      <td style="text-align: center; border: none;">974</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2021-08</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/mbpp">https://huggingface.co/datasets/mbpp</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">MBPP+</td>
      <td style="text-align: center; border: none;">378</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2023-05</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/evalplus/mbppplus">https://huggingface.co/datasets/evalplus/mbppplus</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">CoNaLa</td>
      <td style="text-align: center; border: none;">596.88K</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2018-05</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/neulab/conala">https://huggingface.co/datasets/neulab/conala</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">Spider</td>
      <td style="text-align: center; border: none;">8,034</td>
      <td style="text-align: center; border: none;">SQL</td>
      <td style="text-align: center; border: none;">2018-09</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/xlangai/spider">https://huggingface.co/datasets/xlangai/spider</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">CONCODE</td>
      <td style="text-align: center; border: none;">104K</td>
      <td style="text-align: center; border: none;">Java</td>
      <td style="text-align: center; border: none;">2018-08</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/AhmedSSoliman/CodeXGLUE-CONCODE">https://huggingface.co/datasets/AhmedSSoliman/CONCOD</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">ODEX</td>
      <td style="text-align: center; border: none;">945</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2022-12</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/neulab/odex">https://huggingface.co/datasets/neulab/odex</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">CoderEval</td>
      <td style="text-align: center; border: none;">460</td>
      <td style="text-align: center; border: none;">Python, Java</td>
      <td style="text-align: center; border: none;">2023-02</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/CoderEval/CoderEval">https://github.com/CoderEval/CoderEval</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">ReCode</td>
      <td style="text-align: center; border: none;">1,138</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2022-12</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/amazon-science/recode">https://github.com/amazon-science/recode</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">StudentEval</td>
      <td style="text-align: center; border: none;">1,749</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2023-06</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/wellesley-easel/StudentEval">https://huggingface.co/datasets/wellesley-easel/StudentEval</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">BigCodeBench</td>
      <td style="text-align: center; border: none;">1,140</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2024-06</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/bigcode/bigcodebench">https://huggingface.co/datasets/bigcode/bigcodebench</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">ClassEval</td>
      <td style="text-align: center; border: none;">100</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2023-08</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/FudanSELab/ClassEval">https://huggingface.co/datasets/FudanSELab/ClassEval</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">NaturalCodeBench</td>
      <td style="text-align: center; border: none;">402</td>
      <td style="text-align: center; border: none;">Python, Java</td>
      <td style="text-align: center; border: none;">2024-05</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/THUDM/NaturalCodeBench">https://github.com/THUDM/NaturalCodeBench</a></td>
    </tr>
    <tr style="border-top: 1px solid #333;">
      <td rowspan="3" style="text-align: center; border: none;">Competitions</td>
      <td style="text-align: center; border: none;">APPS</td>
      <td style="text-align: center; border: none;">10,000</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2021-05</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/codeparrot/apps">https://huggingface.co/datasets/codeparrot/apps</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">CodeContests</td>
      <td style="text-align: center; border: none;">13,610</td>
      <td style="text-align: center; border: none;">C++, Python, Java</td>
      <td style="text-align: center; border: none;">2022-02</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/deepmind/code_contests">https://huggingface.co/datasets/deepmind/code_contests</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">LiveCodeBench</td>
      <td style="text-align: center; border: none;">713 Updating</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2024-03</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/LiveCodeBench/LiveCodeBench">https://github.com/LiveCodeBench/LiveCodeBench</a></td>
    </tr>
    <tr style="border-top: 1px solid #333;">
      <td rowspan="3" style="text-align: center; border: none;">Data Science</td>
      <td style="text-align: center; border: none;">DSP</td>
      <td style="text-align: center; border: none;">1,119</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2022-01</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/microsoft/DataScienceProblems">https://github.com/microsoft/DataScienceProblems</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">DS-1000</td>
      <td style="text-align: center; border: none;">1,000</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2022-11</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/xlangai/DS-1000">https://huggingface.co/datasets/xlangai/DS-1000</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">ExeDS</td>
      <td style="text-align: center; border: none;">534</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2022-11</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/Jun-jie-Huang/ExeDS">https://github.com/Jun-jie-Huang/ExeDS</a></td>
    </tr>
    <tr style="border-top: 1px solid #333;">
      <td rowspan="5" style="text-align: center; border: none;">Multilingual</td>
      <td style="text-align: center; border: none;">MBXP</td>
      <td style="text-align: center; border: none;">12.4K</td>
      <td style="text-align: center; border: none;">13</td>
      <td style="text-align: center; border: none;">2022-10</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/mxeval/mbxp">https://huggingface.co/datasets/mxeval/mbxp</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">Multilingual HumanEval</td>
      <td style="text-align: center; border: none;">1.9K</td>
      <td style="text-align: center; border: none;">12</td>
      <td style="text-align: center; border: none;">2022-10</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/mxeval/multi-humaneval">https://huggingface.co/datasets/mxeval/multi-humaneval</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">HumanEval-X</td>
      <td style="text-align: center; border: none;">820</td>
      <td style="text-align: center; border: none;">Python, C++, Java, JavaScript, Go</td>
      <td style="text-align: center; border: none;">2023-03</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/THUDM/humaneval-x">https://huggingface.co/datasets/THUDM/humaneval-x</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">MultiPL-E</td>
      <td style="text-align: center; border: none;">161</td>
      <td style="text-align: center; border: none;">18</td>
      <td style="text-align: center; border: none;">2022-08</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/nuprl/MultiPL-E">https://huggingface.co/datasets/nuprl/MultiPL-E</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">xCodeEval</td>
      <td style="text-align: center; border: none;">5.5M</td>
      <td style="text-align: center; border: none;">11</td>
      <td style="text-align: center; border: none;">2023-03</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/ntunlp/xCodeEval">https://github.com/ntunlp/xCodeEval</a></td>
    </tr>
    <tr style="border-top: 1px solid #333;">
      <td rowspan="5" style="text-align: center; border: none;">Reasoning</td>
      <td style="text-align: center; border: none;">MathQA-X</td>
      <td style="text-align: center; border: none;">5.6K</td>
      <td style="text-align: center; border: none;">Python, Java, JavaScript</td>
      <td style="text-align: center; border: none;">2022-10</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/mxeval/mathqa-x">https://huggingface.co/datasets/mxeval/mathqa-x</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">MathQA-Python</td>
      <td style="text-align: center; border: none;">23,914</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2021-08</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/google-research/google-research">https://github.com/google-research/google-research</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">GSM8K</td>
      <td style="text-align: center; border: none;">8.5K</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2021-10</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/gsm8k">https://huggingface.co/datasets/gsm8k</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">GSM-HARD</td>
      <td style="text-align: center; border: none;">1.32K</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2022-11</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/reasoning-machines/gsm-hard">https://huggingface.co/datasets/reasoning-machines/gsm-hard</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">CRUXEval</td>
      <td style="text-align: center; border: none;">800</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2024-01</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/cruxeval-org/cruxeval">https://huggingface.co/datasets/cruxeval-org/cruxeval</a></td>
    </tr>
    <tr style="border-top: 1px solid #333;">
      <td rowspan="7" style="text-align: center; border: none;">Repository</td>
      <td style="text-align: center; border: none;">RepoEval</td>
      <td style="text-align: center; border: none;">3,573</td>
      <td style="text-align: center; border: none;">Python, Java</td>
      <td style="text-align: center; border: none;">2023-03</td>
      <td style="text-align: center; border: none;"><a href="https://paperswithcode.com/dataset/repoeval">https://paperswithcode.com/dataset/repoeval</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">Stack-Repo</td>
      <td style="text-align: center; border: none;">200</td>
      <td style="text-align: center; border: none;">Java</td>
      <td style="text-align: center; border: none;">2023-06</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/RepoFusion/Stack-Repo">https://huggingface.co/datasets/RepoFusion/Stack-Repo</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">Repobench</td>
      <td style="text-align: center; border: none;">27k</td>
      <td style="text-align: center; border: none;">Python, Java</td>
      <td style="text-align: center; border: none;">2023-01</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/Leolty/repobench">https://github.com/Leolty/repobench</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">EvoCodeBench</td>
      <td style="text-align: center; border: none;">275</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2024-03</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/LJ0815/EvoCodeBench">https://huggingface.co/datasets/LJ0815/EvoCodeBench</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">SWE-bench</td>
      <td style="text-align: center; border: none;">2,294</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2023-10</td>
      <td style="text-align: center; border: none;"><a href="https://huggingface.co/datasets/princeton-nlp/SWE-bench">https://huggingface.co/datasets/princeton-nlp/SWE-bench</a></td>
    </tr>
    <tr>
      <td style="text-align: center; border: none;">CrossCodeEval</td>
      <td style="text-align: center; border: none;">10K</td>
      <td style="text-align: center; border: none;">Python, Java, TypeScript, C#</td>
      <td style="text-align: center; border: none;">2023-10</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/amazon-science/cceval">https://github.com/amazon-science/cceval</a></td>
    </tr>
    <tr style="border-bottom: 1px solid #333;">
      <td style="text-align: center; border: none;">SketchEval</td>
      <td style="text-align: center; border: none;">20,355</td>
      <td style="text-align: center; border: none;">Python</td>
      <td style="text-align: center; border: none;">2024-03</td>
      <td style="text-align: center; border: none;"><a href="https://github.com/nl2code/codes">https://github.com/nl2code/codes</a></td>
    </tr>
  </tbody>
</table>
<p style="text-align: center;">Table.6 测试大语言模型的基准测试集</p>

#### 问题二：大语言模型的幻觉
##### 论文阅读：[Hallucination is Inevitable:An Innate Limitation of Large Language Models](https://arxiv.org/abs/2401.11817)
**幻觉(hallucation)** 指的是大语言模型生成与事实不符、逻辑断裂或者脱离上下文的内容，这一词用以描述AI系统在缺乏真实认知能力的情况下产生的虚构输出。从技术本质来看，幻觉源于模型基于统计模式而非真实世界知识进行文本生成的内在机制，其本质是统计概率驱动的“合理猜测”。
从模型生成的结果与现实割裂的角度来讲，大语言模型的幻觉主要可以分为**事实性幻觉**与**忠实性幻觉**两种，其中事实性幻觉指的是模型生成的内容与可验证的现实世界事实不一致，忠实性幻觉指的是模型生成的内容与用户指令或上下文不一致的情况。

<table>
  <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
    <td style="text-align: center; border: none;"></td>
    <th style="text-align: center; border: none;">回答</th>
    <th style="text-align: center; border: none;">分析</th>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">事实性幻觉</th>
    <td style="text-align: center; border: none;">是的，蜂蜜是天然的，可以帮助糖尿病患者稳定血糖水平。
    </td>
    <td style="text-align: center; border: none;"><strong>错误</strong>：蜂蜜虽然是天然食品，但仍然含有大量果糖和葡萄糖，会升高血糖水平，不适合糖尿病患者代替糖使用。</td>
  </tr>
  <tr style="border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;">忠实性幻觉</th>
    <td style="text-align: center; border: none;">蜂蜜富含维生素和矿物质，对提高免疫力很有帮助，因此是一种健康的食品。</td>
    <td style="text-align: center; border: none;"><strong>偏题</strong>：回答内容虽无事实错误，但与提问“糖尿病患者是否可以用蜂蜜代替糖”无关，未忠实于用户意图。</td>
  </tr>
</table>
<p style="text-align: center;">Table.7 两种幻觉的示例</p>

##### 幻觉高发的场景
<table>
  <tr style="border-top: 1px solid #333; border-bottom: 1px solid #333;">
    <th style="text-align: center; border: none;">场景类别</th>
    <th style="text-align: center; border: none;">具体场景</th>
    <th style="text-align: center; border: none;">示例</th>
    <th style="text-align: center; border: none;">风险等级</th>
    <th style="text-align: center; border: none;">防护建议</th>
  </tr>
  <tr>
    <th rowspan="2" style="text-align: center; border: none;">知识边界模糊</td>
    <td style="text-align: center; border: none;">开放域生成</td>
    <td style="text-align: center; border: none;">续写未完结的经典文学作品</td>
    <td style="text-align: center; border: none;">高</td>
    <td style="text-align: center; border: none;">添加创作范围限制+事实性标注</td>
  </tr>
  <tr>
    <td style="text-align: center; border: none;">未来事件预测</td>
    <td style="text-align: center; border: none;">预测2030年科技突破细节</td>
    <td style="text-align: center; border: none;">极高</td>
    <td style="text-align: center; border: none;">声明预测性质+概率分布呈现</td>
  </tr>
  <tr>
    <th rowspan="2"  style="text-align: center; border: none;">复杂推理</td>
    <td style="text-align: center; border: none;">多跳推理任务</td>
    <td style="text-align: center; border: none;">追溯企业高管早期职业轨迹</td>
    <td style="text-align: center; border: none;">高</td>
    <td style="text-align: center; border: none;">分步验证+外部知识库检索</td>
  </tr>
  <tr>
    <td style="text-align: center; border: none;">数学证明延伸</td>
    <td style="text-align: center; border: none;">要求证明未解决的数学猜想</td>
    <td style="text-align: center; border: none;">极高</td>
    <td style="text-align: center; border: none;">中断机制+当前研究进展说明</td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">技术性诱发</td>
    <td style="text-align: center; border: none;">长文本生成</td>
    <td style="text-align: center; border: none;">小说连续章节生成</td>
    <td style="text-align: center; border: none;">中</td>
    <td style="text-align: center; border: none;">阶段一致性检查+人物属性维护</td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">情感驱动</td>
    <td style="text-align: center; border: none;">安慰性回应</td>
    <td style="text-align: center; border: none;">重症患者寻求治疗方案建议</td>
    <td style="text-align: center; border: none;">极高</td>
    <td style="text-align: center; border: none;">情感剥离响应+理论应用提示</td>
  </tr>
  <tr>
    <th rowspan="3" style="text-align: center; border: none;">特殊领域</td>
    <td style="text-align: center; border: none;">医疗诊断</td>
    <td style="text-align: center; border: none;">根据症状描述提供诊断建议</td>
    <td style="text-align: center; border: none;">极高</td>
    <td style="text-align: center; border: none;">明确非专业建议+医疗数据库</td>
  </tr>
  <tr>
    <td style="text-align: center; border: none;">法律咨询</td>
    <td style="text-align: center; border: none;">解释特定法条适用范围</td>
    <td style="text-align: center; border: none;">高</td>
    <td style="text-align: center; border: none;">司法辖区限定+法律条文引用</td>
  </tr>
  <tr style="border-bottom: 1px solid #333;">
    <td style="text-align: center; border: none;">金融预测</td>
    <td style="text-align: center; border: none;">给出具体股票买卖建议</td>
    <td style="text-align: center; border: none;">极高</td>
    <td style="text-align: center; border: none;">风险提示+历史回报率说</td>
  </tr>
</table>
<p style="text-align: center;">Table.8 幻觉高发的场景</p>

##### 推理能力对幻觉的影响
以Deepseek V3到Deepseek R1中为例，大语言模型解决问题范式从“提问$\rightarrow$回答”到“提问$\rightarrow$思维链$\rightarrow$回答”，推理能力强的模型能减少因逻辑错误导致的幻觉。例如，在数学问题中，模型若具备多步推理能力，更可能得出正确结论而非臆测答案，此外强大的推理能力使模型更精准地捕捉上下文关联，避免因断章取义而生成虚构内容。例如，在问答任务中，模型能通过推理排除干扰选项，降低错误率。但另一方面，低推理能力模型更易回答“不知道”，高推理模型会生成符合概率分布的“自信错误”答案，另一方面来讲如果初始假设错误，但模型基于此展开正确推理也可能导致幻觉发生，综上，**推理能力对幻觉发生的作用是双向的**。

##### 幻觉的处理方法
对于大语言模型的使用者/用户，一般有以下方法应对AI幻觉：
1. 使用联网功能或RAG，为大语言模型提供背景知识
2. 双AI验证，使用多个大模型进行审查，相互监察
3. 通过约束降低虚构可能性
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_05.png" alt="Fig.1-1.5"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-1.1 检索增强代码生成（RACG）的工作流程示意图。当接收到查询（指令）时，检索器会从大规模向量数据库中筛选出相关上下文。随后，检索到的上下文与查询指令合并，该组合输入被送入生成器（大语言模型）以产生目标代码解决方案。</p>

从**约束大模型**的角度出发，可以从以下几个方面限定：
1. 时间锚定：通过规定知识/时间的范围，规避未来时态虚构
2. 知识锚定：声明基于的知识背景，限定权威知识的来源
3. 领域身份：在背景（API中的System）中限定大语言模型的专业身份限定
4. 置信度声明：要求大语言模型标注不确定性的信息，奸杀绝对化错误的断言
5. 上下文提示：类似于RAG，提供外部知识嵌入权威数据片段

从**对抗性提示**的角度出发，有以下的几个方法：
1. 植入反幻觉检测机制： "请用以下格式回答：
- 主要答案（严格基于公开可验证信息）
- 反事实检查 部分（列出可能导致此答案错误的3种假设）“
2. 预设验证条件，迫使模型交叉检查信息：“请先回答“量子纠缠能否证明灵魂存在？”，然后从以下角度验证答案的可靠性： 
- 物理学界主流观点； 
- 近五年相关论文数量； 
- 是否存在可重复实验证据。 
3. 链式验证：请完成以下验证链：
- 陈述观点：______ 
- 列出支撑该观点的三个权威数据源 
- 检查每个数据源是否存在矛盾信息4. 最终结论（标注可信度等。

#### 问题二：推理中失效的解决方案
##### 1.问题重述
在评估大语言模型（LLMs）的能力时，研究者通常关注其文本生成、问答、代码合成等高层次任务的表现。然而，一些对人类而言极其简单的任务——如字母计数和数值比较——却可能让最先进的模型频繁出错。这两个案例具有典型性，能够揭示LLMs在符号处理和结构化推理方面的根本性局限。

**案例1**：字母计数（如 "strawberry" 的字母数）
- 人类可以轻松拆解单词（s-t-r-a-w-b-e-r-r-y）并计数（10个字母）。而模型会出现由于混淆语义相关词的长度、忽略重复字母（如漏计双"r"）等问题，导致计数任务的失败。
  
**案例2**：数值比较（如 9.8 vs 9.11）
- 人类可以自动理解小数位值（9.8 = 9.80 > 9.11）。而模型可能会由于字符串字典序错误判断、混淆整数与小数逻辑或对数值尺度不敏感等原因，给出9.11 > 9.8的结论，造成大小判断任务的失败。
##### 2.问题意义
这两类任务看似简单，但能有效揭示LLMs的核心局限：

- 符号操作的精确性不足（字母计数依赖严格的字符级处理）。
- 结构化推理能力缺失（数值比较需要位值理解，而非单纯模式匹配）。
- 训练数据与目标错位（预训练优化语义关联，而非符号逻辑）。
  
已有研究表明（Saxton et al., 2019），在算术推理和符号操作任务上，纯神经语言模型的性能远低于混合架构（如神经符号系统）。因此，这两个案例不仅是技术问题，更指向LLMs的认知架构缺陷。

##### 3.现有解决方案

当前针对大语言模型在字母计数和数值比较等符号推理任务上的缺陷，研究者提出了多种解决方案。最直接的方法是让模型调用外部工具，比如集成Wolfram Alpha或Python解释器来处理精确计算，这种方法虽然能实现接近100%的准确率，但需要额外的基础设施支持，并会引入200-500毫秒的延迟。另一种思路是让模型生成可执行代码来解决这些问题，例如自动编写字符计数函数或数值比较脚本，在沙盒环境中运行，实验显示这种方法能达到98%以上的准确率，但存在代码生成错误和安全风险。

思维链提示技术通过引导模型分步推理来提升表现，比如让模型先拆解字母再计数，或逐步比较数字的每一位。这种方法无需外部依赖，在7B参数模型上就能将字母计数准确率从42%提升到78%，但对复杂任务仍存在中间步骤错误的问题。更先进的工具调用范式（如Toolformer）让模型学会自主判断何时调用计算器、搜索引擎等外部工具，通过预训练时的特殊标记学习，将数值比较准确率从88.7%提升到99.5%，但需要精细的API管理和调用策略。

最新研究如Meta的符号记忆模块和Google的CALM架构显示，将符号处理能力直接内置到模型中是更有潜力的方向，能在不依赖外部系统的情况下使字母计数准确率达到99.4%。不过，对于特殊字符、科学计数法等情况，现有方案仍存在不足。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/Toolformer.png" alt="Fig.3-2.1"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-2.1 模型自主决定调用不同的API（从上至下依次为：问答系统、计算器、机器翻译系统和维基百科搜索引擎）来获取有助于完成文本片段的有用信息</p>

综上，现有在**不改变神经网络基本架构的基础上**的解决方法主要有：
- 大语言模型融合外部解决器（如OpenAI和walframe的合作）
- 大语言模型生成解决问题的脚本代码而直接生成结果
- 使用思维链辅助推理
- 停用提示词产生对应格式的输出

#### 自动化代码生成的智能体

要理解自动优化代码生成的智能体技术，就需要首先理解大语言模型作为代码评判者的技术。大型语言模型强大的指令遵循能力激发了研究人员对基于大型语言模型的评估潜力的创新性研究。大模型作为评判者是指将先进的专有大型语言模型（例如，GPT4、Gemini 和 Claud 3）用作人类评估者的代理，这涉及设计包含特定要求的提示。这种方法减少了对人工参与的依赖，从而促进了更高效、可扩展的评估。此外，大型语言模型可以为分配的评分提供有见地的解释，从而增强评估的可解释性。
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/LLM4J.png" alt="Fig.3-1.2"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-1.2 代码大语言模型作为评估者（LLM-as-a-judge）对代码大模型生成内容进行评测的流程框架。主要存在两种方法：成对比较（pairwise comparison）与单答案评分（single answer grading）</p>

利用这种技术可以实现大语言模型对代码的自我改进，具体流程如下图3-1.3所示：

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/LLM_SC.png" alt="Fig.3-1.3"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-1.3 使用提示词实现大语言模型自我改进的代码生成流程示意图。该过程通过整合执行结果进行迭代式自我优化，并包含可选的自反思机制以提升生成质量。</p>

最近，除了单独使用前文中讨论的思维链技术以外，一些研究提出使用多智能体写作来提高基于LLM的代码生成的有效性和准确性，其中每个智能体处理一个独特任务，例如代码生成或任务规划等。这些多智能体写作框架旨在通过分配工作负载和优化代码生成过程中各个方面的性能来克服单智能体方法的局限性。

<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/s1_07.png" alt="Fig.3-1.4"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-1.4 基于大语言模型的自主智能体系统通用架构。规划模块：智能体将复杂任务拆解为可管理的子目标，或通过自我批判与行为反思从过往错误中学习以优化后续表现。记忆模块：该组件使智能体能够存储和检索历史信息。工具模块：智能体被训练调用外部函数或应用程序接口。行动模块：无论是否借助工具，智能体通过执行动作与环境进行交互。图中灰色虚线表示系统内部的数据流向。</p>

在自动化代码生成领域，基于大语言模型（LLM）的自主智能体已展现出卓越能力。例如，AgentCoder在HumanEval基准测试中实现了96.3%的突破性通过率，向自动化软件开发的未来迈进重要一步。AgentCoder克服了传统多智能体方法缺乏有效的反馈机制以及涉及大量token资源在大量智能体之间的通信和协调的问题。
<img src="https://github.com/Anorexia16/PicStream/releases/download/asd4/AgentCoder.png" alt="Fig.3-1.5"
style="display: block; margin: 0 auto;">
<p style="text-align: center;">Fig.3-1.5 AgentCoder 的流程，以及来自 HumanEval 的代码生成示例</p>

AgentCoder的框架及其流程如图3-1.3所示。该过程首先将任务/代码生成需求/描述输入到代码生成代理（Agent#1：程序员代理）。随后，测试用例生成器（Agent#2：测试设计者代理）的任务是生成测试用例，这些测试用例用于评估程序员代理生成的代码片段的正确性。代码片段和测试用例由测试执行代理（Agent#3）收集，并在本地环境（本地终端）中执行，以获得反馈（即，代码是否通过所有测试，以及代码在某些测试中失败时的错误消息）。如果测试执行代理发现代码片段通过所有测试用例，它将把代码返回给用户并完成迭代。否则，测试执行代理会将测试执行错误消息返回给程序员代理。然后迭代继续，程序员代理重新生成代码片段以解决反馈中发现的问题，测试执行代理重新执行新代码并向程序员代理提供新的反馈，直到测试执行代理发现代码通过所有测试。
以编码任务“检查给定数字列表中是否存在任意两个数字之间的距离小于给定阈值（如图3.-1.3所示）”为例，在初始代码生成过程中，程序员代理将尝试理解和澄清给定的任务，在本例中，即解释识别列表中彼此之间在指定阈值范围内的数字对的要求。程序员代理随后将决定解决该问题的算法或方法。这可能涉及选择一种有效的方法来比较列表中每对数字。接下来，在伪代码创建过程中，程序员代理将为解决方案开发一个循序渐进的指南或伪代码，确保操作的逻辑流程。最后，在代码生成阶段，程序员会将伪代码翻译成可执行代码。

此外，代码自动生成多智能体也有其他实现。例如，创新性元编程框架MetaGPT将人类工作流效率融入基于LLM的多智能体协作；CodeAct使用可执行Python代码统一LLM智能体动作空间，而非生成JSON或文本格式指令；最新提出的AutoCodeRover能自主解决GitHub问题以实现程序优化升级。

## **总结**  

#### **大语言模型在算法设计的应用**  
大语言模型（LLMs）在算法设计中展现出多样化角色与潜力。作为**优化器**，LLMs可通过迭代生成并改进解决方案，解决组合优化、多目标优化等问题；作为**预测器**，其能基于历史数据预测算法性能，降低计算成本；作为**提取器**，LLMs从复杂描述中提炼关键特征，增强算法适配性；作为**设计者**，则直接生成算法逻辑或组件代码，显著提升开发效率。结合搜索方法（如进化搜索、不确定性引导搜索）和提示策略（如思维链、自一致性），LLMs在优化、机器学习、科学发现及工业领域广泛应用。然而，其在可扩展性、可解释性、领域知识融合等方面仍面临挑战。  

#### **运用大语言模型的推理能力提高代码生成的准确率**  
通过结构化推理策略可有效提升代码生成质量。**思维链（CoT）** 分步拆解需求，引导模型模拟人类逻辑；**结构化思维链（SCoT）** 进一步引入顺序、分支、循环等编程结构，作为自然语言与代码间的桥梁，在HumanEval等基准测试中Pass@1指标最高提升13.79%。**ReAct框架** 结合推理（Reason）与行动（Act），通过交互式验证减少幻觉错误，增强生成代码的可靠性。实验表明，融合推理策略的模型生成的代码更符合开发者习惯，代码坏味道减少36.08%，且对示例扰动具有更强鲁棒性。  

#### **当前研究进展**  
1. **算法设计系统化**：LLM4AD领域已形成多维度分类体系，涵盖角色定位、搜索方法、提示策略及应用场景，为算法自动化设计提供理论框架。  
2. **推理能力增强**：CoT、SCoT等提示策略显著提升多步任务性能，ReAct等混合方法推动复杂任务（如交互式代码生成）的突破。  
3. **代码生成优化**：模型如DeepSeek-V3通过稀疏注意力机制和层次化表示学习，在长代码生成任务中保持效率；DeepSeek-R1结合强化学习优化推理路径，降低错误传播风险。  
4. **问题暴露与改进**：研究揭示了LLMs在符号推理（如字母计数、数值比较）中的固有缺陷，并提出工具调用、神经符号融合等解决方案，部分任务准确率可达99%以上。  

#### **未来研究方向**  
1. **可解释性与可靠性**：开发可视化工具解释模型决策逻辑，引入多轮验证和自动化测试确保代码安全性与功能正确性。  
2. **领域知识深度集成**：构建领域专属LLMs，融合知识图谱与专业数据，提升医疗、金融等场景的算法生成专业性。  
3. **多模态与跨任务迁移**：探索文本、图像、API文档的多模态输入，支持跨语言（如C++/Python）代码生成与优化。  
4. **低资源与高效计算**：研究模型压缩、联邦学习技术，降低训练与推理成本，推动设备部署。  
5. **认知架构创新**：结合神经符号计算，内置符号处理模块以解决结构化推理缺陷，突破当前黑箱生成局限。  

