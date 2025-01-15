# Sound and Complete Witnesses for Template-Based Verification of LTL Properties on Polynomial Programs

>多项式程序LTL特性基于模板验证的健全完整见证人
>
>作者：Krishnendu Chatterjee、Amir Goharshady、Ehsan Goharshady、Mehrdad Karrabi和Ðorđe Žikelić
>
>单位： **Institute of Science and Technology Austria (ISTA)**,**The Hong Kong University of Science and Technology (HKUST)**,**Singapore Management University**
>
>会议： FM 2024
>
>论文链接： [Sound and Complete Witnesses for Template-Based Verification of LTL Properties on Polynomial Programs](https://arxiv.org/pdf/2403.05386)

| Recorder | Date       |
| -------- | ---------- |
| 王颖     | 2025-01-03 |

# 一、研究背景

## 1. LTL 在程序验证中的重要性

线性时态逻辑（LTL）是形式规范、模型检查和程序验证的经典框架。它能表达如终止性、活性、公平性和安全性等常见验证任务，通过原子命题和无限轨迹来描述程序的性质。

## 2. 见证（Witnesses）的意义

在验证中，给定规范 \$\(\phi\) \$和程序 \(P\)，见证是一种数学对象，其存在可证明规范满足程序。对于不可判定的验证问题，虽然问题的一般情况不可判定，但合理且完整的见证概念可引导出检查特殊形式见证存在性的算法。

## 3. 多项式程序的研究价值

许多现实世界的程序（如网络 - 物理系统和智能合约程序）可在多项式算术程序框架下建模。多项式程序在可判定性和通用性之间提供了较好的权衡，并且通过抽象解释或数值逼近，其分析可应用于许多非多项式程序。此外，之前对线性/仿射程序和多项式程序在特定规范下的研究较多，LTL 涵盖这些规范，对多项式程序的 LTL 分析可统一之前的工作。

---

# 二、研究方向

## 1. 理论方面

通过探索与 Büchi 自动机的联系，为一般 LTL 公式提出合理且完整的见证族，扩展并统一已知的排序函数、归纳可达性见证和归纳不变量等概念。

## 2. 算法方面

针对多项式程序，提出一种合理且半完整的基于模板的算法来合成多项式 LTL 见证，这是对之前考虑终止性、可达性和安全性的基于模板方法的推广。

## 3. 实验方面

实现所提出的方法，并与最先进的 LTL 模型检查工具进行比较，以展示该方法在实践中的有效性。

---

# 三、研究方法

## 1. 见证定义

### 存在性 B - PA 见证（EBRF）

对于转换系统 \$T=(V,L,l_{init},\theta_{init},\to)\$ 和状态集 \$B\subseteq S\$ 中的 EB - PA 问题，EBRF 是一个函数 \$(f:S\to R)\$，满足存在初始状态 \$s_{init}\in S_{init}\$ 使得 \$f(s_{init})\geq0\$，且对于每个 \$s_1\in S\$，存在 \$s_2\in S\$ 使得 \$s_1\to s_2\$ 且 \$(s_1,s_2)\$ 被 \$f\$进行 Büchi - 排序。

### 普遍性 B - PA 见证（UBRF）

函数 \$f:S\to R^n\$ 满足对于每个 \$s\in S_{init}\$，\$f(s)\geq0\$，且对于每个 \$s_1,s_2\in S\$ 使得 \$s_1\to s_2\$，\$(s_1,s_2)\$ 被 \$f\$ 进行 Büchi - 排序。

## 2. 基于模板的多项式见证合成算法

### 多项式 EBRF 合成算法

#### 步骤一：固定符号模板

在每个位置 \$l\in L\$ 为 EBRF 生成符号多项式模板 \$f_l(x)=\sum_{i = 1}^{k}c_{l,i}\cdot m_i\$，其中 \$c\$ - 变量为表示多项式系数的符号模板变量。

#### 步骤二：生成蕴含约束

对于每个位置 \$l\in L\$ 和变量估值 \$x\models\theta_l\$，必须存在一个输出转换 \$\tau\$ 使得 \$x\models G_{\tau}\$ 且 \$\tau\$ 被 \$f\$ 在 \$x\$ 中进行 Büchi - 排序，将此条件符号性地写为蕴含约束 \$forall x\in R^n,x\models(\varphi_l\Rightarrow\psi_l)\$，并对 \$\psi_l\$ 进行处理以便后续操作。

#### 步骤三：将约束简化为二次不等式

将每个形式为 \$\Phi\Rightarrow\Psi\$ 的约束，先将 \$\Phi\$ 写为析取范式，\$\Psi\$ 写为合取范式，然后使用 Putinar’s Positivstellensatz 生成一组等价的二次不等式。

#### 步骤四：处理初始条件

为每个变量 \$x\in V\$ 引入符号模板变量 \$t_x\$，模拟程序中 \$x\$ 的初始值，并添加约束 \$[\theta_{init}(t)\wedge f_{l_{init}}(t)\geq0]\$ 到二次规划 \$\Gamma\$。

#### 步骤五：求解系统

使用外部求解器（如 SMT 求解器）计算 \$t\$ 和 \$c\$ 变量的值，若求解成功则得到 EBRF 的具体实例，否则返回“未知”。

---

# 四、研究结果

## 1. 理论结果

证明了 EBRFs 对于 EB - PA 问题的合理性和完整性，以及 UBRFs 对于 UB - PA 问题的合理性和完整性。

## 2. 算法结果

多项式 EBRF 和 UBRF 合成算法是合理且半完整的归约到二次规划的算法。

## 3. 实验结果

![](C:\Users\wying\Desktop\Snipaste_2025-01-15_18-18-52.png)

### 线性程序结果

在与 Ultimate LTLAutomizer、nuXmv、MuVal 以及直接应用 Z3（不使用 Putinar’s Positivstellensatz）的对比实验中，在大多数情况下，该工具在证明和反驳 LTL 公式方面表现优于直接应用 Z3 的方法，与 Ultimate 和 MuVal 相当且能证明 10 个独特实例，在部分情况下比 nuXmv 证明更多实例。在运行时间方面，该工具和 Ultimate 在证明 LTL 验证实例时速度最快，在 LTL 反驳方面该工具比其他工具慢。

### 非线性程序结果

在非线性基准测试中，与 nuXmv 和 MuVal 对比，该工具在除一个公式外的所有公式中成功解决的实例更多，且能证明 11 个其他工具无法处理的实例，而 Ultimate 不支持非线性算术，Z3 在每个基准测试中都超时，进一步证明了算法中步骤三（量词消去过程）的实际必要性。

---

# 五、研究结论

1. 提出了基于模板的 LTL 验证的合理且完整的见证族，适用于程序中 LTL 性质的验证和反驳，统一并推广了之前针对 LTL 特殊情况的工作。
2. 证明了 LTL 见证可通过归约到二次规划以合理且半完整的方式合成（当程序和见证都是多项式时）。
3. 未来可考虑允许堆操作的非数值程序，将处理堆操作的方法与本方法相结合是一个有吸引力的研究方向。

---

# 六、研究不足

1. 方法局限于多项式程序和见证，对于需要非多项式见证的底层程序可能会失败，例如 Ultimate LTLAutomizer 能处理非多项式程序和见证，而本方法不能。
2. 在 LTL 反驳方面，该工具比其他工具慢。

---

# 七、未来研究方向

1. 考虑允许堆操作的非数值程序，将处理堆操作的方法（如构建数值抽象）与本方法相结合。
