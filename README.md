# Academic_Blog
形式化方法与软件工程相关前沿工作阅读与分享

## 一、引言

在快速发展的软件工程领域中，确保软件系统的正确性和安全性已成为不可忽视的关键问题。形式化验证作为一种严谨的数学方法，通过构建软件的数学模型并应用逻辑推理来验证系统是否满足预期规范，正逐渐成为保障软件质量的重要手段。本简报旨在提供一个全面的形式化验证概览，并洞悉当前国际上的最新研究动态。

### 二、形式化验证基础

定义：形式化验证是一种基于数学逻辑的方法，用于严格证明软件系统（或硬件设计）是否满足其规格说明，即是否“做正确的事”。

形式化验证的金字塔：
- 上近似：尝试覆盖所有可能的行为，确保无遗漏，但可能导致验证过程复杂且耗时。
- 下近似：简化系统模型，仅验证关键路径或常见情况，提高效率但可能遗漏错误。
- 人类辅助：在高度自动化的验证过程中，人类专家仍扮演重要角色，特别是在定义规格、解读验证结果及调试复杂证明时。

![](/images/top/Advance_001.png)

### 三、Six Schools of Verification

- Static Analysis：静态分析通过不执行程序来检查代码中的潜在错误，如类型错误、未初始化变量等。
- Abstract Interpretation：抽象解释通过构建程序的抽象模型来近似分析其行为，适用于性能优化和安全性检查。
- Testing & Symbolic Execution：测试结合符号执行，通过具体输入和符号输入探索程序路径，发现缺陷。
- Model Checking：模型检查通过穷举系统状态空间来验证性质，适用于有限状态系统的正确性验证。
- Deductive Verification：演绎验证基于逻辑证明，从公理出发逐步推导出系统满足特定性质的结论。
- Functional Verification：功能验证关注于验证软件功能是否符合预期，通常结合形式化规格说明。

![](/images/top/Advance_002.png)

### 四、国际顶级会议前沿动态

- 软件工程领域：ICSE、ASE、FSE等会议持续关注软件质量保障、自动化测试与验证技术的新进展。
- 程序设计语言领域：POPL、PLDI、OOPSLA等探讨了编程语言设计如何支持更高效、更安全的形式化验证。
计算机理论和形式化方法领域：

#### FM‘24亮点文章：
- “A Pyramid Of (Formal) Software Verification”综述了形式化验证的层次结构与挑战。
- “Accelerated Bounded Model Checking”提出了加速模型检查的新算法，提高了验证效率。
- “Accurate Static Data Race Detection for C”介绍了针对C语言的高效静态数据竞争检测方法。
- “Reachability Analysis for Multiloop Programs Using Transition Power Abstraction”提出了多循环程序可达性分析的新方法。

#### CAV‘24亮点文章：
- “Verification Algorithms for Automated Separation Logic Verifiers”研究了自动化分离逻辑验证器的算法。
- “Framework for Debugging Automated Program Verification Proofs via Proof Actions”构建了一个调试自动化验证证明的框架。
- “Interactive Theorem Proving modulo Fuzzing”结合了交互式定理证明与模糊测试，提升了验证的实用性和效率。

### 五、结语
形式化验证作为软件工程领域的前沿技术，正不断向着更高效、更自动化的方向发展。通过持续关注国际顶级会议的研究成果，我们可以及时把握技术脉搏，推动形式化验证技术在实践中的广泛应用。期待在未来的快报中，能够深入剖析更多创新技术与成功案例，共同推动软件工程领域的进步。