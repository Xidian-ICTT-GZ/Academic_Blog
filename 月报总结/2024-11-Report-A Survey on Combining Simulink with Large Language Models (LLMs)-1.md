# Simulink结合大语言模型（LLMs）的相关研究工作调研（上）

|Recorder|Date|Categories|
|----|----|----|
|Tang YeXin，Su ZheXin，Li ChunYi, Ma Zhi|2024-11-30|Formal Modeling |

-----

## 引言

### 调研背景

Simulink是MATLAB提供的强大工具，广泛应用于动态系统的建模、仿真与设计。其模块化建模方式和多领域支持使其成为工业界和学术界不可或缺的系统设计平台。

与此同时，大语言模型（LLM, Large Language Models），如OpenAI的GPT系列和Google的BERT等，以其自然语言处理的卓越能力正在逐步影响各行业。其在语言生成、知识推理、代码生成等领域的表现，展示了跨领域协作的无限可能。

Simulink与大语言模型结合，有望通过自然语言驱动的建模、仿真与优化，进一步提升系统设计的效率和创新能力。这种结合不仅能为工程设计带来变革，也能为智能系统开发提供新的技术框架。

### 调研目的

本次调研旨在探讨Simulink与大语言模型结合的可能性，分析技术结合点、典型应用场景与技术挑战，并展望这一新兴领域的未来发展。

## Simulink概述

Simulink是由MathWorks公司开发的多领域动态系统建模和仿真工具，广泛应用于控制系统、信号处理、通信系统和嵌入式系统开发中。其主要特点包括：

- 图形化建模：基于模块的拖拽式操作，简化了复杂系统的建模过程。
- 多领域支持：支持物理建模、信号流建模和多领域交叉建模。
- 实时仿真：支持与硬件连接进行实时仿真。
- 自动化代码生成：支持将模型转换为C/C++或HDL代码，用于嵌入式系统。

### Simulink 的典型应用：

1. 汽车工业：自动驾驶与动力系统仿真。
2. 航空航天：飞行器控制和任务规划。
3. 能源行业：智能电网仿真和新能源系统优化。

![image](https://github.com/user-attachments/assets/111e680b-bc0d-4384-9be6-33bc7d616b55)

图源：知乎用户@王之葵托利

尽管Simulink强大且灵活，但在模型开发过程中仍存在挑战，如模型构建的复杂性、模块复用的困难、错误调试成本高等问题。

## 大语言模型（LLMs）概述

### 语言模型历史

![image](https://github.com/user-attachments/assets/6b0476a3-8a42-44a6-a98d-1e961ef7646a)

图源：同济大学《自然语言处理前沿——大语言模型的前世今生》

大语言模型（如OpenAI的GPT系列、Google的Bard）是基于深度学习的自然语言处理系统，其核心技术包括Transformer架构和大规模预训练。这些模型通过学习海量数据中的语言模式，具备以下特点：

- 强大的语言生成与理解能力：能够进行上下文对话、内容创作、问题回答等。
- 跨领域适应性：能应用于翻译、编程、文本分析等不同任务。
- 增强的推理能力：在特定情况下能够进行逻辑推导和任务规划。

### 应用现状：

- 代码生成：如 GitHub Copilot，辅助开发者完成代码编写。
- 自然语言接口：通过简单指令生成复杂的程序或脚本。
- 跨领域辅助：自动驾驶、天文学数据分析等。

## Simulink与大语言模型结合的潜力与应用场景

### 自动化建模与设计支持

大语言模型可以将自然语言描述转化为Simulink模型框架。例如，用户提供“构建一个PID控制器”，模型能够自动生成对应的模块。

在复杂系统建模中，帮助设计初始框架，从而降低手工建模的工作量。

### 相关论文

#### Requirements-driven Slicing of Simulink Models Using LLMs

链接：[2405.01695](https://arxiv.org/abs/2405.01695) (arxiv.org)

##### 一、研究背景

传统的模型切片技术依赖于手动创建的追溯链接，这些链接可能会耗时且容易出错。大型语言模型的出现提供了一种可能，即不依赖手动追溯链接就能自动化切片过程。本论文的研究重点在于利用大语言模型（LLMs）从Simulink模型中自动提取基于需求的切片。

##### 二、基础知识

链式思考：鼓励模型不仅生成最终的答案，而且逐步展示出它是如何推理并得出结论的。在执行复杂问题求解时，模型会生成一系列中间步骤，每个步骤都可以视为解答问题的一个逻辑片段或计算过程的一部分。

零样本学习：一种能够在没有任何样本的情况下学习新类别的方法。通常情况下，模型只能识别它在训练集中见过的类别。但通过零样本学习，模型能够利用一些辅助信息来进行推理，并推广到从未见过的类别上。这些辅助信息可以是关于类别的语义描述、属性或其他先验知识。

![image](https://github.com/user-attachments/assets/baae2c06-0d58-47f6-8921-daaee2031bce)

图源：CSDN@羽林小王子

##### 三、研究目的

作者提出一种基于大型语言模型的方法，用于从图形化的Simulink模型中提取模型切片，这样可以实现自动化地从Simulink模型中提取与特定需求相关的小型切片，简化了验证与调试过程，确保关键需求的正确性。

##### 四、实验方法

![image](https://github.com/user-attachments/assets/e70383b0-42e4-419a-9e94-753c1333f1af)

五个输入：

Simulink模型<br>
计算模型切片的自然语言需求陈述R<br> 
用于提取与R相关的Simulink块的提示模板<br>
训练示例<br>
文本表示粒度的冗长程度

##### 五、实验过程

**Step1** 将Simulink模型转换为文本表示，以便LLM的处理。这里作者进行实验探讨了不同粒度的转换对生成切片的准确性的影响。

模型节选分别在高中低三种冗余度下的文本表示：

![image](https://github.com/user-attachments/assets/f4befa1c-d367-4982-9377-a8c46b5bc215)

**Step2** 作者设计了不同的提示策略（思维链策略、N-shot 策略和Z-shot策略），引导LLM识别满足特定需求的Simulink块。

![image](https://github.com/user-attachments/assets/af63737a-9d8c-4f2a-a43a-699ca3c1531a)

**Step3** 根据LLM识别的模块，构建模型切片。通过执行这些切片并比较它们满足需求的能力，与原始模型进行对比，以验证切片的准确性。如果在评估R的适配性时，在切片和原始模型上都产生正值，或者都产生负值/零值，就认为根据要求R生成的切片是准确的。

##### 六、实验结果分析及结论

![image](https://github.com/user-attachments/assets/6257f68d-701d-499c-95f6-c9cd2499b9d3)

“H”、“M”和“L”分别表示高冗余度、中冗余度和低冗余度；

“CT”、“NS”和“ZS”分别对应思维链策略、N-shot 策略和Z-shot策略。

每列对应Tustin模型的要求R，即 R1-R5。

✓表示切片和原始模型的适配性极性相同；

✗表示切片的适配性极性与原始模型不同；

V表示切片完全满足要求。

![image](https://github.com/user-attachments/assets/bbee9360-71a0-4ac8-9570-cb51dc06aa5f)

-表示不准确的模型切片

由表3可得，低冗余度的配置生成的无效切片最多。这表明高度抽象化（即低冗余度）的模型缺乏必要的细节，导致LLM无法很好地识别需求。同时，高冗余度配置在模型地文字描述中加入了视觉渲染细节，这可能会导致LLM产生“幻觉”，无法处理所有细节以精准定位相关区块，从而产生错误。

实验结果表明，保留Simulink块的语法和语义信息，同时省略视觉渲染信息的文本表示，能够产生最准确的切片。结合链式思考和零样本提示策略，中等粒度的转换能够产生最准确的模型切片（即M-CT与M-ZS）。

##### 七、技术挑战与未来研究方向

1.大语言模型有上下文输入限制，处理复杂Simulink模型时具有较大的挑战。

2.将Simulink模型文本化时，如何平衡详细信息与简洁性的需求是一个很重要的问题。当信息趋于详细时，大语言模型难以处理所有细节进行精准定位；而当信息过于抽象简洁时，大语言模型又将由于缺乏必要的细节无法很好地识别需求，所以如何对模型的语义和语法进行恰当地保留是一个不小的技术挑战。

3.该研究中使用的训练和测试模型有限。在实验中只使用了一个训练和一个测试的Simulink模型，同时也只使用了Chatgpt这一大语言模型。使用不同的Simulink模型或者使用不同的大语言模型可能会影响识别准确性，未来还需要进一步扩大评估范围，采用更多的实验进行验证。

4.在未来的实验中，可以考虑做Simulink-LLM接口设计，使Simulink模型的转换与LLM的提示生成无缝集成。

### 优化与自动调试

利用LLMs分析Simulink模型，建议改进信号传递、优化参数设置，甚至自动修复模型错误。

### 相关论文

#### SLGPT: Using Transfer Learning to Directly Generate Simulink Model Files and Find Bugs in the Simulink Toolchain

链接：[SLGPT](https://arxiv.org/abs/SLGPT) (arxiv.org)

##### 一、研究背景

由于Simulink代码库庞大且缺乏完整的形式化语言规范，发现错误非常困难。深度学习技术可以从样本模型中学习语言规范，但需要大量训练数据才能有效工作。

##### 二、基础知识

迁移学习

目标：将某个领域或任务上学习到的知识或模式应用到不同但相关的领域或问题中。

主要思想：从相关领域中迁移标注数据或者知识结构、完成或改进目标领域或任务的学习效果。

关键点：

![image](https://github.com/user-attachments/assets/50e1f9cc-04f6-45ab-a18a-0bf2c16c8873)

图源：https://blog.csdn.net/dakenz/article/details/85954548

##### 三、方法提出与问题解决

作者提出SLGPT（使用迁移学习直接生成Simulink模型文件并发现Simulink工具链中的错误），利用预训练的GPT-2模型，通过迁移学习适应Simulink，以生成更高质量的Simulink模型并发现更多的工具链错误。

##### 四、实验过程

**Step1** 数据收集：通过随机模型生成器SLforge和开源代码库（GitHub和MATLAB Central）获取Simulink模型。

![image](https://github.com/user-attachments/assets/ae2d88a1-7cf5-498d-8ec3-73fffc051037)

**Step2** 数据预处理：简化模型以去除冗余信息（如注释、默认配置、块位置信息），并采用广度优先算法重构模型以适应GPT-2的学习风格。

![image](https://github.com/user-attachments/assets/a8224a71-d9ba-406f-bb53-bbb0258ecba7)

重构算法解析：<br>
source_blks (S)：Simulink模型的起始块集合，代表模型中无输入的起始块。<br>
other_blks (B)：模型中剩余的其他块。<br>
graph_info (G)：描述Simulink模型的图信息，包括块之间的连线关系（邻接信息）。<br>
C_BFS：重新排列后的Simulink模型块序列（基于广度优先搜索顺序）。

首先当起始块集合S或其余块集合B非空时，进入主循环，初始化空队列Q。主循环用来处理集合S和B：从S或B中取出一个块b，优先从S中选取（如果S非空），将b添加到队列Q的尾部。接着采用广度优先算法进行队列的处理，当队列Q为空时，回到主循环，检查S和B是否仍有未处理的块。如果两者都为空，算法终止。

**Step3** 模型训练与生成：使用预训练的GPT-2模型，并用随机生成的模型和开源模型进行微调，然后从调整过的GPT-2模型中迭代采样生成Simulink模型文件。

**Step4** 模型验证：使用有效性检查器检测Simulink工具的崩溃，并手动审查每个崩溃案例。

##### 五、实验结果与评估

生成模型的有效性（RQ1）：
SLGPT生成的模型编译成功率为43%-47%，而DeepFuzzSL的成功率为42%-90%（取决于训练数据源）。SLGPT生成的模型在结构特性上更接近开源模型，如连通性更强、输入输出端口的连接更合理。

缺陷发现能力（RQ2）：
SLGPT生成的模型发现了DeepFuzzSL发现的所有已知缺陷，并新增发现了一些未被记录的缺陷。

![image](https://github.com/user-attachments/assets/1d948359-95cd-47a3-b538-1705e3c6324a)

第一行、第二行和第三行分别为训练模型、DeepFuzzSL生成的模型、SLGPT生成的模型的实验结果。
第一列、第二列、第三列和第四列分别为每个模型中的块数量、连接的子图数量、最大连接子图中的块数量和最大路径长度。
与最接近的竞争对手相比，SLGPT能够生成有效且更接近开源模型的Simulink模型，并且发现了比DeepFuzzSL更多的Simulink开发工具链错误。同时，它的实现、参数设置和训练集都是开源的。


##### 六、个人观点

尽管SLGPT通过迁移学习解决了训练数据不足的问题，但仍然需要更大、更多样化的数据集以进一步提高模型的泛化能力。同时，因为Simulink模型较为复杂，且没有公开可用的完整规范，所以仍需要进一步研究以减少模型生成过程中出现的错误。

### 测试与验证自动化

#### 测试用例生成与智能验证

根据模型功能和设计目标，LLM自动生成全面的测试用例，确保模型在各种极端条件下的可靠性。分析测试结果，诊断失败原因并提出改进方案，提升系统的安全性和稳定性。

#### 相关论文

**论文1**：[An Empirical Study of Using Large Language Models for Unit Test Generation (arxiv.org)](https://arxiv.org/pdf/2305.00418v3)

**论文2**：[[2310.02368] Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation (arxiv.org)](https://arxiv.org/pdf/2310.02368)

论文1重点研究了大语言模型在单元测试生成中的应用，这启示我们可以探索类似的方法，提供Simulink模型的结构、属性或功能性描述作为输入，从而生成相关的测试、验证模型，甚至代码。

论文2重点研究了利用强化学习和静态质量度量优化生成高质量的单元测试代码。作者提出利用静态质量指标，例如语法正确性、断言的存在性、调用焦点方法等为测试代码生成提供反馈，通过引入类似的静态分析工具，可以为Simulink的模型生成建立质量标准，评估自动生成的模型的有效性与正确性。同时，论文中提到的静态质量度量强化学习框架通过基于奖励信号优化生成过程，展示了如何让大语言模型生成更符合质量要求的代码，我们可以尝试将这一方法延伸到Simulink中，利用强化学习优化模型生成任务。例如，可以通过设计奖励机制，鼓励生成更优结构的模型，减少模型中的冗余或潜在错误。

上述两篇文章分别介绍了LLM生成测试用例的有效性验证以及微调方法。在学术论文搜索引擎上检索LLM测试用例生成时，从论文发表的时间和数量能明显感觉到LLM在单元测试领域处于起步阶段。但不可否认的是，它已经表现出了巨大的潜力。

结合以上两篇论文可以看出当前评价测试用例的好坏主要是编译率、测试通过率、覆盖率等静态质量指标。评价的测试集有两种，一组是HumanEval，另一组是开源库。前者的指标明显高于后者。当前SFT是主流的微调方法，但强化学习也开始融入微调的过程中，能够有效提高模型的性能。不过每一种语言对于测试用例都有各自的测试框架，在目前的大部分研究中，主要是以单语言来进行训练，并未考虑多语言的情况，所以在未来的研究中，可以对这方面进行更深入的探索。

### 当前研究进展与案例

#### LLM驱动的代码生成与解释

已有研究表明，LLMs可将自然语言翻译为MATLAB代码，而MATLAB代码可进一步用于Simulink建模。例如，OpenAI的Codex模型被用于将控制系统的文字描述转化为MATLAB脚本，再通过脚本自动生成Simulink模型。

#### 相关论文

链接：[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (arxiv.org)

论文详细描述了Codex模型的能力，其中包括将自然语言描述转换为代码的能力，Codex生成的MATLAB代码可以进一步用于Simulink建模。

## 结论

Simulink和大语言模型结合可以带来巨大的技术价值和应用前景，这种结合不仅可以提高建模效率、优化仿真质量，还可以推动自动化设计流程的发展。然而，目前仍需进一步研究以克服数据接口、模型可靠性和适用性等技术挑战。未来，随着工具链和技术的成熟，这一结合有望成为工程仿真领域的关键创新方向。

## 参考文献

1. Luitel D, Nejati S, Sabetzadeh M. Requirements-driven Slicing of Simulink Models Using LLMs[J]. arXiv preprint arXiv:2405.01695,2024.
2. Shrestha S L, Csallner C. SLGPT: Using transfer learning to directly generate Simulink model files and find bugs in the Simulink toolchain[C]//Proceedings of the 25th International Conference on Evaluation and Assessment in Software Engineering. 2021: 260-265.
3. Siddiqa M L, Santos J C S, Tanvirb R H, et al. An Empirical Study of Using Large Language Models for Unit Test Generation[J]. arXiv preprint arXiv:2305.00418, 2023.
4. Steenhoek B, Tufano M, Sundaresan N, et al. Reinforcement learning from automatic feedback for high-quality unit test generation[J]. arXiv preprint arXiv:2310.02368, 2023.
5. Chen M, Tworek J, Jun H, et al. Evaluating large language models trained on code[J]. arXiv preprint arXiv:2107.03374, 2021.
