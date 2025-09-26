# 软件工程与形式化方法相关前沿工作阅读与分享

### 一、引言

在快速发展的软件工程和形式化方法领域，确保软件系统的可信性一直都是不可忽视的关键问题。本简报旨在提供一个较为全面的软件测试验证方法概览，并洞悉当前国际上的最新研究动态。在此，我们汇聚课题组师生之智，共同细品国际顶级会议论文的字里行间，在思维的碰撞中激发新的火花，让灵感在交流与思辨中璀璨绽放。

我们重点关注以下会议的最新研究工作：

- 软件工程：[ICSE](https://conf.researchr.org/home/icse-2025)、[FSE](https://conf.researchr.org/home/ase-2025)、[ASE](https://conf.researchr.org/home/ase-2025)、[ISSTA](https://conf.researchr.org/home/issta-2025)、[SOSP](https://sigops.org/s/conferences/sosp/2025/index.html)、[OSDI](https://www.usenix.org/conference/osdi26)
- 程序设计语言：[POPL](https://conf.researchr.org/home/POPL-2025)、[PLDI](https://conf.researchr.org/home/PLDI-2025)、[OOPSLA](https://2025.splashcon.org/track/OOPSLA)
- 计算机理论与形式化方法：[CAV](https://conferences.i-cav.org/2025/)、[FM](https://www.fm24.polimi.it/)
- 计算机安全：[S&amp;P](https://sp2025.ieee-security.org/)、[CCS](https://www.sigsac.org/ccs/CCS2025/)、[Usenix Security](https://www.usenix.org/conference/usenixsecurity25)、[NDSS](https://www.ndss-symposium.org/ndss2025/)
- 人工智能与自然语言处理：[AAAI](https://aaai.org/conference/aaai/aaai-26/)、[IJCAI](https://2025.ijcai.org/)、[NeurIPS](https://neurips.cc/Conferences/2025)、[ICML](https://icml.cc/Conferences/2025)、[ACL](https://2025.aclweb.org/)、[EMNLP](https://2025.emnlp.org/)

### 二、程序测试验证基本概念

程序验证，作为确保软件质量与安全的核心环节，广义上的概念就是根据既定的标准（又被称之为规范--Specification），对被检查事物（通常是程序--Program），进行严谨的属性审查与确认的过程。程序验证的主要输入是规范和程序，下图描述了一个基本的程序验证过程。这一过程不仅仰赖于计算资源的支持，更离不开工程师的智慧投入。

![Alt](/images/top/verification_process.png#pic_center)

**程序--Program**：作为验证活动的核心对象，其形态万千，不仅涵盖了C/C++、Java、Rust等传统或新兴编程语言编写的软件，还延伸至多线程程序、中断驱动系统、智能合约、深度神经网络，乃至硬件设计、并行与分布式软件，以及协议、进程演算、自动机、物理信息融合系统等更为抽象的计算模型。

**规范--Specification**：作为检查程序是否满足标准的准则集合，承载着对程序应展现行为的精确描述与期望。这些规范可能涉及功能正确性、安全性、鲁棒性、性能等多维度的性质，是验证工作据以评判程序是否达标的准绳。从形式化规约语言到自然语言描述，规范的多样性与灵活性为验证工作带来了既丰富又复杂的挑战。

**验证结果--Result**：理想的结果是系统验证通过（Pass），但也有可能检测到一个或多个缺陷（Fail），或者验证没有定论，最终结果未知（Unknown）。从实际的角度来看，未知可能是最糟糕的结果，因为它意味着辛勤付出的努力却未能收获确凿的结论。然而，正如我们将看到的，可靠地避免未知的结果是非常具有挑战性的。

**金字塔上的策略、权衡与攀登之路**：一个理想的验证过程应该能够自动进行，无需人工干预；能够准确无误地识别出所有不符合规范的问题，即不遗漏任何错误；同时，也能避免误报，只在系统确实满足规范时给出“验证通过”的结论。然而，现实世界中并不存在这样一种万能的验证方法和工具，它能在有限时间内对所有程序和规格进行完美验证，既无遗漏又无误判。

为了直观地理解这一挑战，我们可以将验证方法的特性构想为一个三棱金字塔模型（从顶部俯瞰），如下图所示。金字塔的三个底边分别代表了自动性（Automatic）、无遗漏错误（No Missed Bugs，即完全正确性）和无误报（No False Alarms,即精确性）这三个核心属性。金字塔的顶点，则是我们梦寐以求的理想验证系统，它集自动性、完全正确性和精确性于一身。

![Alt](/images/top/pyramid.png#pic_center)

金字塔的底面，则分布着三种不同的策略，它们分别对应着三个属性的极端表现：

- 上近似（Over-approximation）：这种方法倾向于保守地估计程序的行为，可能会误报一些实际上符合规范的情况，以确保不遗漏任何潜在的错误。它牺牲了精确性以换取完全正确性。
- 下近似（Under-approximation）：与上近似相反，下近似方法更加谨慎地确认程序的正确性，只报告那些确实违反规范的情况，从而避免了误报。然而，这种谨慎可能导致一些错误被遗漏，因此它牺牲了完全正确性以换取精确性。
- 人工辅助（Human-assisted）：这种方法结合了人的智慧和机器的自动化能力，通过人工参与来弥补自动化工具的不足。虽然这种方法可能增加了验证过程的复杂性和时间成本，但它能够在一定程度上平衡自动性、完全正确性和精确性之间的冲突。

攀登这座金字塔，即是在追求验证方法的不断完善和优化。不同的验证技术和工具，如同不同的攀登路线，它们根据自身的特点和优势，在自动性、完全正确性和精确性这三个维度上做出不同的权衡和取舍。计算资源的投入，就像攀登过程中的体力消耗，是推动我们不断接近理想顶点的动力。

这个金字塔模型虽然并非面面俱到，除了上述三个核心属性外，软件验证技术还可以根据其他多个维度进行分类和比较。例如，属性的指定方式、被分析系统的类型、验证系统和规范语言的易用性等，都是影响验证方法选择和效果的重要因素。但这个金字塔模型为我们提供了一个有用的起点，帮助我们在面对验证问题做出初步的决策和规划，在阅读前沿论文的时候了解方法的定位和局限性。在攀登金字塔的过程中，我们需要不断探索和尝试新的验证技术和方法，以找到最适合当前问题和资源的解决方案。同时，我们也需要保持对验证目标的清晰认识和对验证过程的严格控制，以确保我们的每一步都朝着理想状态迈进。

### 三、程序测试验证六大类方法

程序测试验证存在着多种不同的学术传统和方法，它们各自形成了独特的思想体系、方法和社区。尽管这些方法之间存在互动、重叠和相互启发，但每一种都拥有独特的理论基础、技术手段和应用场景。以下表格对程序验证的六大类方法进行了简要概述，它们共同构成了软件验证的丰富体系。

![Alt](/images/top/six_school.png#pic_center)

- **静态分析（Static Analysis）：**
  - 特点：静态分析是一种在不执行程序的情况下，通过检查程序代码来发现潜在错误、安全漏洞和性能问题的方法。
  - 验证对象：用于各种类型的程序，特别是那些对安全性和可靠性要求较高的软件。静态分析能够识别出代码中的语法错误、类型不匹配、未初始化的变量使用、内存泄漏等常见问题。
  - 技术要点：包括语法分析、语义分析、数据流分析、控制流分析等，以及相应的数学形式化方法来表达分析结果。
  - 工具示例：常见的静态分析工具包括Clang Static Analyzer、Cppcheck、FindBugs等。
- **抽象解释（Abstract Interpretation）：**
  - 特点：抽象解释是一种基于格理论的程序分析方法，它通过构建程序的抽象模型来近似地表示程序的行为，并在此基础上进行分析和验证。这种方法允许开发者在保持程序关键性质的同时，忽略掉不重要的细节，从而简化分析过程。
  - 验证对象：适用于需要精确控制流和数据流信息的程序分析任务。
  - 技术要点：涉及抽象域的选择、抽象操作的定义、格结构的构建等关键技术。
  - 工具示例：抽象解释通常与其他分析技术结合使用，如与静态分析或模型检查相结合。
- **测试和符号执行（Testing and Symbolic Execution）：**
  - 特点：测试是验证程序功能正确性的直接方法，它通过实际执行程序并观察输出结果来发现错误。而符号执行则是一种自动化的测试技术，它使用符号表示程序的输入和状态，通过模拟程序执行来探索所有可能的执行路径。
  - 验证对象：广泛适用于各种类型的程序，特别是那些难以通过形式化方法验证的复杂系统。
  - 技术要点：包括测试用例的设计、执行和评估，以及符号执行的路径探索、约束求解等。
  - 工具示例：JUnit（用于Java程序的单元测试）、KLEE（一个符号执行引擎）等。
- **模型检查（Model Checking）：**
  - 特点：模型检查是一种自动化的验证技术，它通过遍历程序的状态空间来检查是否满足给定的规范或性质。
  - 验证对象：这种方法特别适用于具有有限状态空间的程序，如硬件电路、通信协议等。模型检查能够发现程序中的逻辑错误、死锁、活锁等问题。
  - 技术要点：包括状态空间的表示、遍历算法的选择、性质的定义和检查等。
  - 工具示例：SPIN、NuSMV、CBMC等是知名的模型检查工具。
- **演绎验证（Deductive Verification）：**
  - 特点：演绎验证是一种基于数学逻辑和定理证明的验证方法。它从程序的规范出发，通过逻辑推导来证明程序满足特定的性质或要求。
  - 验证对象：适用于那些可以精确描述其规范和性质的程序，如安全关键系统。
  - 技术要点：包括逻辑系统的选择、定理证明的策略、证明自动化等。演绎验证能够提供严格的数学保证，是验证安全关键系统和高可靠性软件的首选方法。然而，由于它需要高度的数学技能和复杂的证明过程，因此在实际应用中具有一定的挑战性。
  - 工具示例：Frama-C、Dafny等是常用的演绎验证工具。
- **功能验证（Functional Verification）：**
  - 特点：功能验证主要关注程序的功能正确性，即程序是否按照预期的方式执行并产生正确的结果。
  - 验证对象：广泛适用于各种类型的程序，特别是那些对功能正确性要求较高的软件。
  - 技术要点：包括功能测试、等价性检查、模拟与仿真等。
  - 工具示例：Isabelle/HOL、Coq等是常用的功能验证工具。

这六大类方法各有优劣，适用于不同的验证场景和需求。在实际应用中，开发者通常会根据项目的具体情况和目标来选择合适的方法或组合多种方法进行综合验证。同时，随着技术的不断进步和研究的深入，新的验证方法和技术也在不断涌现，为软件验证领域带来了更多的可能性和挑战。

### 四、每月简报
- [2025年5月简报：大语言模型在形式化数学定理证明方向的调研](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2025-05-Report-A%20Survey%20on%20Formal%20proof%20of%20mathematical%20theorems%20via%20LLM.md)
- [2025年4月简报：大语言模型在代码生成相关方向的调研（下）](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2025-04-Report-A%20Survey%20on%20Large%20Language%20Model%20in%20the%20directions%20of%20code%20generation%20(2).md)
- [2025年3月简报：大语言模型在代码生成相关方向的调研（上）](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2025-03-Report-A%20Survey%20on%20Large%20Language%20Model%20in%20the%20directions%20of%20code%20generation%20(1).md)
- [2025年1月简报：深度神经网络验证与测试的相关研究工作调研](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2025-01-Report-A%20Survey%20on%20Verification%20and%20Testing%20of%20Deep%20Neural%20Networks%20(DNNs).md)
- [2024年12月简报：Simulink结合大语言模型（LLMs）的相关研究工作调研（下）](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2024-12-Report-A%20Survey%20on%20Combining%20Simulink%20with%20Large%20Language%20Models%20(LLMs)-2.md)
- [2024年11月简报：Simulink结合大语言模型（LLMs）的相关研究工作调研（上）](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2024-11-Report-A%20Survey%20on%20Combining%20Simulink%20with%20Large%20Language%20Models%20(LLMs)-1.md)
- [2024年10月简报：面向C/C++程序的地址消毒器性能优化洞察分析与实证研究报告](/%E6%9C%88%E6%8A%A5%E6%80%BB%E7%BB%93/2024-10-Report-Performance%20Optimization%20of%20AddressSanitizer.md)

### 五、最新前沿论文分享

##### 2025
- [2025-09-26-QREI-An Innovative Heuristic to Detect Special States in Concurrent Software Systems](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-09-26-QREI-An%20Innovative%20Heuristic%20to%20Detect%20Special%20States%20in%20Concurrent%20Software%20Systems.pdf)
- [2025-09-26-Tamgram: A Frontend for Large-scale Protocol Modeling in Tamarin](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-09-26-Tamgram%20A%20Frontend%20for%20Large-scale%20Protocol%20Modeling%20in%20Tamarin.pdf)
- [2025-09-26-SOSP23-Snowcat: Efficient Kernel Concurrency Testing using a Learned Coverage Predictor](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-09-26-SOSP23-Snowcat%20Efficient%20Kernel%20Concurrency%20Testing%20using%20a%20Learned%20Coverage%20Predictor.pdf)
- [2025-09-18-ASPLOS20-Effective Concurrency Testing for Distributed Systems](/2025阅读笔记/2025-09-18-ASPLOS20-Effective%20Concurrency%20Testing%20for%20Distributed%20Systems.pdf)
- [2025-09-18-ThreadSanitizer data race detection in practice](/2025阅读笔记/2025-09-18-ThreadSanitizer%20data%20race%20detection%20in%20practice.pdf)
- [2025-09-18-ReForm — Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs A Preliminary Study on Dafny](/2025阅读笔记/2025-09-18-ReForm%20—%20Reducing%20Human%20Priors%20in%20Scalable%20Formal%20Software%20Verification%20with%20RL%20in%20LLMs%20A%20Preliminary%20Study%20on%20Dafny.pdf)
- [2025-09-11-usenixsec25-RangeSanitizer: Detecting Memory Errors with Efficient Range Checks](/2025阅读笔记/2025-09-11-usenixsec25-RangeSanitizer%20Detecting%20Memory%20Errors%20with%20Efficient%20Range%20Checks.pdf)
- [2025-09-11-NeurIPS25-FVEL Interactive Formal Verification Environment with Large Language Models via Theorem Proving](/2025阅读笔记/2025-09-11-NeurIPS25-FVEL%20Interactive%20Formal%20Verification%20Environment%20with%20Large%20Language%20Models%20via%20Theorem%20Proving.pdf)
- [2025-09-11-OOPSLA25-Laurel: Unblocking Automated Verification with Large Language Models](/2025阅读笔记/2025-09-11-OOPSLA25-Laurel%20Unblocking%20Automated%20Verification%20with%20Large%20Language%20Models.pdf)
- [2025-09-11-TOSEM-Structured Chain-of-Thought Prompting for Code Generation](/2025阅读笔记/2025-09-11-TOSEM-Structured%20Chain-of-Thought%20Prompting%20for%20Code%20Generation.pdf)
- [2025-09-04-FSE25-CXXCraffer: An LLM-Based Agent for Automated C/C++ Open Source Software Building](/2025阅读笔记/2025-09-04-FSE25-CXXCraffer%20An%20LLM-Based%20Agent%20for%20Automated%20CC++%20Open%20Source%20Software%20Building.pdf)
- [2025-09-04-POPL25-Stateless Model Checking Concurrent/Distributed Programs](/2025阅读笔记/2025-09-04-popl25-Stateless%20Model%20Checking%20Concurrent:Distributed%20Programs.pdf)
- [2025-03-20-NeurIPS24-Towards General Loop Invariant Generation: A Benchmark of Programs with Memory Manipulation](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-03-20-program%20verification%20with%20LLM.pdf)
- [2025-03-20-Enhancing Automated Loop Invariant Generation for Complex Programs with Large Language Models](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-03-20-program%20verification%20with%20LLM.pdf)
- [2025-03-20-S&P25-Poster: Enhancing Symbolic Execution with LLMs for Vulnerability Detection](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-03-20-S%26P25-Poster%20Enhancing%20Symbolic%20Execution%20with%20LLMs%20for%20Vulnerability%20Detection.pdf)
- [2025-03-14-Retrieval-Augmented Generation (RAG): Paradigms, Technologies, and Trends](https://zhuanlan.zhihu.com/p/28884173045)
- [2025-03-14-Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://blog.csdn.net/DEVELOPERAA/article/details/145302836)
- [2025-03-07-Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving](https://zhuanlan.zhihu.com/p/25278606872)
- [2025-02-27-DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Searc](https://deepseek.csdn.net/67d623d56670175f993701ae.html)
- [2025-02-27-PLDI09-SoftBound: highly compatible and complete spatial memory safety for c](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-02-27-PLDI09-Nagarakatte_SoftBound.pdf)
- [2025-01-17-S&amp;P25-Evaluating the Effectiveness of Memory Safety Sanitizers](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-01-17-S&P25-Evaluating%20the%20Effectiveness%20of%20Memory%20Safety%20Sanitizers.md)
- [2025-01-17-ICPP-W 2023-Enhanced Memory Corruption Detection in CC++ Programs](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-01-17-ICPP-W%202023-Enhanced%20Memory%20Corruption%20Detection%20in%20CC++%20Programs.pdf)
- [2025-01-03-ASE24-LLM Meets Bounded Model Checking Neuro-symbolic Loop Invariant Inference](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-01-03-ASE24-LLM%20Meets%20Bounded%20Model%20Checking%20Neuro-symbolic%20Loop%20Invariant%20Inference.pdf)
- [2025-01-03-NeurIPS24-Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-01-03-NeurIPS24-Scaling%20LLM%20Test-Time%20Compute%20Optimally%20can%20be%20More%20Effective%20than%20Scaling%20Model%20Parameters.pdf)
- [2025-01-03-FM24-Sound and Complete Witnesses for Template-Based Verification of LTL Properties on Polynomial Programs](/2025%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2025-01-03-FM24-Sound%20and%20Complete%20Witnesses%20for%20Template-Based%20Verification%20of%20LTL%20Properties%20on%20Polynomial%20Programs.md)

##### 2024
- [2024-12-22-FM24-Reachability Analysis for Multiloop Programs Using Transition Power Abstraction](https://avm2024.informatik.uni-freiburg.de/assets/presentations/konstantin_britikov.pdf)
- [2024-12-22-FM24-Sound and Complete Witnesses for Template-Based Verification of LTL Properties on Polynomial Programs](https://link.springer.com/chapter/10.1007/978-3-031-71162-6_31)
- [2024-12-10-ASPLOS24-GiantSan Efficient Memory Sanitization with Segment Folding](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-12-10-ASPLOS24-GiantSan%20Efficient%20Memory%20Sanitization%20with%20Segment%20Folding.md)
- [2024-12-08-ASPLOS24-Greybox Fuzzing for Concurrency Testing](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-12-08-ASPLOS24-Greybox%20Fuzzing%20for%20Concurrency%20Testing.md)
- [2024-12-05-POPL19-Incorrectness Logic](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-12-05-POPL19-Incorrectness%20Logic.pdf)
- [2024-11-22-ICSE22-Controlled Concurrency Testing via Periodical Scheduling](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-11-22-ICSE22-Controlled%20Concurrency%20Testing%20via%20Periodical%20Scheduling.md)
- [2024-11-22-FSE21-Conditional interpolation making concurrent program verification more effective](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-11-22-FSE21-Conditional%20interpolation%20making%20concurrent%20program%20verification%20more%20effective.md)
- [2024-11-08-SEC24_Pruning Redundant Sanitizer Checks](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-11-08-SEC24_Pruning%20Redundant%20Sanitizer%20Checks.pdf)
- [2024-11-08-SEC23_MTSan_A Feasible and Practical Memory Sanitizer for Fuzzing COTS Binaries](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-11-08-SEC23_MTSan_A%20Feasible%20and%20Practical%20Memory%20Sanitizer%20for%20Fuzzing%20COTS%20Binaries.pdf)
- [2024-10-25-OSDI21_SANRAZOR Reducing Redundant Sanitizer Checks in CC++ Programs](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-10-25-OSDI21_SANRAZOR%20Reducing%20Redundant%20Sanitizer%20Checks%20in%20CC%2B%2B%20Programs.pdf)
- [2024-10-25-ACT20_FuzzSan_Efficient Sanitizer Metadata Design for Fuzzing](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-10-25-ACT20_FuzzSan_Efficient%20Sanitizer%20Metadata%20Design%20for%20Fuzzing.pdf)
- [2024-05-19-TKDD24-LLM4SA Automatically Inspecting Thousands of Static Bug Warnings with Large Language Model](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-05-19-TKDD24-LLM4SA%20Automatically%20Inspecting%20Thousands%20of%20Static%20Bug%20Warnings%20with%20Large%20Language%20Model.md)
- [2024-05-19-ICSE24-RPG Rust Library Fuzzing with Pool-based Fuzz Target Generation and Generic Support](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2024-05-19-ICSE24-RPG%20Rust%20Library%20Fuzzing%20with%20Pool-based%20Fuzz%20Target%20Generation%20and%20Generic%20Support.md)
- [2021-01-22-CCS20-RTFM_Automatic Assumption Discovery and Verification Derivation from Library Document for API Misuse Detection](/2024%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/2021-01-22-CCS20-RTFM_Automatic%20Assumption%20Discovery%20and%20Verification%20Derivation%20from%20Library%20Document%20for%20API%20Misuse%20Detection.md)

### 五、结语

程序测试验证作为软件工程和形式化方法领域的前沿技术，正不断向着更高效、更自动化、更智能的方向发展。通过持续关注国际顶级会议的研究成果，我们可以及时把握技术脉搏，推动形式化验证技术在实践中的广泛应用。期待在每一次的简报中，我们能够不断深入剖析更多创新技术与成功案例，共同推动程序测试验证领域的进步。