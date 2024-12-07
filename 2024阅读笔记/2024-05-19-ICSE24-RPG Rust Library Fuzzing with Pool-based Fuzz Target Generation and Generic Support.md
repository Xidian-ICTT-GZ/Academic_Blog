# RPG: Rust Library Fuzzing with Pool-based Fuzz Target Generation and Generic Support

>RPG：一种面向Rust库的模糊测试目标自动生成技术
> 
>作者：Zhiwu Xu, Bohao Wu, Cheng Wen, Bin Zhang, Shengchao Qin, Mengda He.
>
>单位： Shenzhen University, 
>
>会议： ICSE 2024
>
>论文链接： [RPG: Rust Library Fuzzing with Pool-based Fuzz Target Generation and Generic Support](https://wcventure.github.io/pdf/ICSE2024_RPG.pdf)

|Recorder|Date|Categories|
|----|----|----|
|张彬|2024-05-19|Testing, Rust Program|

## 背景介绍

Rust是一种新兴的系统编程语言，它具有内存安全和高效性的优点。Rust的设计理念是“不信任开发者”，也就是说，它不允许任何可能导致未定义行为的操作。这样，Rust可以避免很多常见的编程错误，比如空指针解引用、缓冲区溢出、内存泄漏等。由于这些优势，Rust不仅受到了很多系统开发者的青睐，也吸引了其他领域的科学家。

Rust库（crate）是Rust语言的基本组成部分，它们是一些可复用的代码模块，可以提供各种功能和特性。Rust库有两种类型：库（library）和二进制（binary）。库可以被其他库或二进制引用，而二进制则是可执行的程序。Rust库可以通过一个叫做[Crates.io](http://Crates.io)的在线仓库进行发布和下载，目前已经有超过6万个Rust库可供使用。然而，Rust的库也不是完美的，它们可能存在一些隐藏的错误和漏洞，这些错误和漏洞可能导致程序崩溃或者出现未定义行为。那么，如何有效地检测和发现Rust库中的错误呢？

一种常用的方法是模糊测试（fuzzing），它是一种自动化的软件测试技术，它通过向程序提供随机或者半随机的输入，观察程序的行为是否异常。模糊测试已经被证明是一种非常有效的发现软件错误的方法，它已经在许多领域和项目中得到了广泛的应用。然而，模糊测试也有一些挑战和限制，其中之一就是如何为目标程序编写合适的测试用例（也称为fuzz target）。对于Rust库来说，这个问题尤为突出，因为Rust库通常只提供了一些函数（也称为API），而没有给出具体的使用场景和示例。因此，如果想要对Rust库进行模糊测试，就需要手动编写fuzz target，这是一项非常耗时和繁琐的工作。

为了解决这个问题，一些研究人员提出了一种自动化的模糊测试目标（fuzz target）生成技术，它可以根据Rust库的API信息，自动地生成一些有效和有效的fuzz target，从而实现对Rust库的模糊测试，例如发表在ASE’24的[RULF](https://ieeexplore.ieee.org/abstract/document/9678813/)。该文章指出，模糊测试目标是一个可以调用库中的函数（API）的Rust程序，它需要满足以下条件：

- 语法正确：模糊测试目标必须是一个合法的Rust程序，可以通过编译器的检查。

- 语义正确：模糊测试目标必须是一个有意义的Rust程序，可以正确地调用库中的函数，不会产生编译时或运行时的错误。

- 多样性：模糊测试目标应该能够调用库中的不同函数，以及函数之间的不同组合，以触发库中的不同分支和路径。

- 有效性：模糊测试目标应该能够调用库中的关键函数，特别是那些涉及到不安全（unsafe）代码的函数，以发现库中的潜在错误。

## 本文方法

RPG是本文提出的一种自动为Rust库生成模糊测试目标的新技术，它的全称是(R)ust (P)ool-based fuzz target (G)eneration。RPG的主要贡献是解决了两个挑战：

- 如何生成多样性和有效性高的函数序列，以优先考虑不安全代码和函数之间的交互，从而揭示Rust库中的错误。

- 如何提供对泛型（generic）函数的支持，并验证模糊测试目标的语法和语义正确性，以达到高覆盖率。

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1715930495755-4cc3d600-bb89-4325-886e-fa813baf4b7a.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_1042%2Climit_0" width="600"></div>

RPG的工作流程如下：

- Rust库分析：RPG利用静态分析从Rust库中提取函数和数据类型的信息，基于此构建一个API依赖图和一个参数提供器。API依赖图记录了库中可以调用的函数（API）的信息，用于指导函数序列的生成；参数提供器包含了库中定义的数据类型以及一些常用的数据结构类型，用于为泛型函数提供类型候选。

- 函数序列生成：这一步是RPG的核心，因为一个函数序列直接反映了其对应的模糊测试目标的质量。RPG从一个序列集合开始，该集合考虑了不安全的API，并根据API依赖图和一个API池生成函数序列。API池由当前可用的API组成，其中可能有多个重复。特别地，当处理泛型函数时，RPG从参数提供器中取出数据类型，并保持泛型类型参数的一致性，以确保有效性。

- 函数序列优化：为了确保模糊测试目标的语法和语义正确性，RPG采用了移动-借用（move-borrow）循环检查和泛型声明检查来移除无效的函数序列。RPG还进行了序列过滤，以获得一个最小的序列集合，覆盖了最多的API及其依赖。最后，剩余的函数序列分别被合成，生成模糊测试目标。

RPG的优势：

- 优先考虑不安全代码和交互：文章的方案能够有效地检测Rust库中由不安全代码或复杂API序列交互导致的漏洞，而过去的方案主要关注单个API的覆盖，忽略了这些重要的因素。

- 提供泛型支持和有效性检查：文章的方案能够处理Rust库中广泛使用的泛型函数，通过类型推断和参数提供器为泛型参数提供合适的类型，并保持类型的一致性。文章的方案还能够对API序列进行移动-借用检查和泛型声明检查，以确保API序列的语法和语义有效性，而过去的方案缺乏对泛型的支持或只提供有限的支持，导致生成的目标不完善或无效。

- 使用基于池的序列生成和过滤：文章的方案能够使用一种基于池的方法，根据API的不安全性和依赖性，生成多样化和深度的API序列，以覆盖更多的API和依赖。文章的方案还能够对API序列进行过滤，以获得一个最小的序列集合，覆盖最多的API和依赖，而过去的方案使用的是基于图的序列生成，不能有效地探索API序列的可能性，也没有对API序列进行过滤，导致生成的目标冗余或低效。

## 实验效果

**任务和性能**：文章在50个流行的Rust库上评估了RPG，并将库按规模归为Micro、Medium、Huge、Small、Unsafe、Large和Generic五类，使用AFL++作为模糊测试工具。文章使用了以下指标来衡量RPG的性能：

- API覆盖率：表示生成的模糊测试目标覆盖了多少比例的可达API。文章的结果显示，RPG的API覆盖率为71.8%，显著高于RULF的48.9%。

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1716098698734-df371082-43c1-4da0-aa32-a056a1f1d4b7.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_757%2Climit_0" width="600"></div>

- 依赖覆盖率：表示生成的模糊测试目标覆盖了多少比例的API依赖。文章的结果显示，RPG的依赖覆盖率为11.1%，显著高于RULF的3.7%。

- 目标数量：表示生成的模糊测试目标的数量。文章的结果显示，RPG的目标数量为1,165，低于RULF的1,481，说明RPG生成的目标更精简和高效。

- 目标有效性：表示生成的模糊测试目标是否能够通过编译和运行。文章的结果显示，RPG的目标有效性为100%，高于RULF的94.9%，说明RPG生成的目标更符合Rust的语法和语义规则。

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1716100472251-fff70221-9021-4f71-9698-94a11545a99f.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0" width="700"></div>

- 漏洞发现能力：表示生成的模糊测试目标能够发现多少个漏洞。文章的结果显示，RPG能够发现25个之前未知的漏洞，高于RULF的8个，说明RPG生成的目标更能够触发Rust库中的隐藏漏洞。

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1716102591462-7aee5b2b-cee3-43ae-8fb4-1e95030370d4.png?x-oss-process=image%2Fformat%2Cwebp" width="500"></div>

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1716102607769-a02dcdcc-0df9-4420-a73f-70459e0d3eb7.png?x-oss-process=image%2Fformat%2Cwebp" width="500"></div>

## Case Study

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1715932922302-0967b766-faf7-415a-8791-e0ac0cbe32ff.png?x-oss-process=image%2Fformat%2Cwebp" width="600"></div>

<div align=center><img src="https://cdn.nlark.com/yuque/0/2024/png/40574035/1715932941832-b9886bb9-b736-4ab6-b35a-44a988b76ca8.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_750%2Climit_0" width="620"></div>

## 局限性讨论

参数提供器的局限性：文章的方案使用了一个参数提供器，包含了Rust库中定义的数据结构类型和一些常用的数据结构类型，作为泛型参数的候选。然而，这个参数提供器可能不能覆盖所有可能的类型，导致一些泛型函数无法被调用。文章的方案也没有考虑用户自定义的类型，可能导致一些泛型函数的调用不符合用户的期望。

类型推断的不完善性：文章的方案使用了一个轻量级的类型推断引擎，根据API依赖和泛型约束，推断泛型参数的具体类型。然而，这个类型推断引擎可能不能处理一些复杂的类型关系，例如多态性，继承，重载等，导致一些泛型函数的调用不正确或失败。

模糊测试的不确定性：文章的方案使用了AFL++作为模糊测试工具，对生成的目标进行模糊测试，以发现漏洞。然而，模糊测试本身是一个不确定的过程，可能受到很多因素的影响。
