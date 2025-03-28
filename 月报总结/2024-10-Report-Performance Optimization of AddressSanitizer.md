# 面向C/C++程序的地址消毒器性能优化洞察分析与实证研究报告

|Recorder|Date|Categories|
|----|----|----|
|林志伟、奚佳新|2024-11-27|Testing, Dynamic Analysis|

-----

## 一、概述

AddressSanitizer（地址消毒器，简称ASan）[1] 是一种内存错误检测工具，目的是帮助开发者检测和调试内存相关的问题，如使用未分配的内存、使用已释放的内存、堆内存溢出等。ASan是由Google开发的，广泛用于C、C++等语言的代码中。通过使用ASan，开发者可以在早期阶段发现和解决潜在的内存错误问题，有效提高程序的稳定性和安全性。ASan可以用于检测各种类型的内存安全错误，支持检查的内存安全错误主要包括：
- Use after free 
- Double free
- Heap buffer overflow
- Stack buffer overflow
- Global buffer overflow
- Use after return
- Use after scope
- Null-pointer Dereference
- Initialization order bugs
- SEGV
- Memory leaks

ASan工具已经作为一个LLVM的Pass，集成至LLVM中，可以通过-fsanitizer=address编译选项开启它。ASan的源码位于/lib/Transforms/Instrumentation/AddressSanitizer.cpp中，Runtime-library的源码在llvm的另一个项目compiler-rt的/lib/asan文件夹中。该工具适用于x86、ARM、MIPS（所有架构的32位和64位版本）、PowerPC64。支持的操作系统有Linux、Darwin（OS X和iOS模拟器）、FreeBSD、Android。

|**OS**	| x86 | x86_64 | ARM | ARM64 | MIPS | MIPS64 |PowerPC |
|----|----|----|----|----|----|----|----|
| Linux | yes | yes | | | yes | yes | yes | yes |
| OS X | yes | yes  | | |     | | |
| iOS Simulator	| yes | yes |
| FreeBSD | yes | yes | 
| Android | yes | yes | yes | yes |

尽管ASan在检测内存安全方面是一个非常有用的工具，但由于其会引入一些性能开销，因此在生产环境中也通常难以应用ASan。本章详细介绍了ASan的基本原理及其问题挑战，就“是否存在一种方法能够有效降低ASan运行时开销与内存开销，以促进ASan在实际生产环境中更大程度的应用”这一研究问题，对近期国内外针对ASan的性能开销进行优化的研究工作进行了调研与归纳总结。


## 二、ASAN的基本原理及其问题挑战

内存错误是导致软件安全问题的主要原因之一，AddressSanitizer（ASan）能够在程序运行时有效检测多种内存错误。从Asan工具的组成上看，该工具由一个编译器插桩模块(Instrumentation)和一个运行时库(Run-time library,例如替换malloc函数等)组成。从Asan技术的基本原理上看，主要包括三个部分：①影子内存（Shadow Memory）+ ②插桩检测（Instrumentation）+ ③红区投毒（Poisoned Redzone）

#### 2.1 影子内存
概念：影子内存是一种内存映射技术，用于记录主内存区域的分配和使用情况。ASAN将进程的虚拟内存空间划分为主应用内存区（Mem）和影子内存区（Shadow）。主应用内存区是普通APP代码内存使用区，而影子内存区仅ASAN感知，与主应用内存区存在一种类似“影子”的对应关系。
实现方式：进程的虚拟内存空间被ASAN划分为2个独立的部分：
- 主应用内存区 (Mem): 普通APP代码内存使用区。
- 影子内存区 (Shadow): 该内存区仅ASAN感知，影子顾名思义是指该内存区与主应用内存区存在一种类似“影子”的对应关系。ASAN在将主内存区的一个字节标记为“中毒”状态时，也会在对应的影子内存区写一个特殊值，该值称为“影子值”。

ASan在将主应用内存区的一个字节标记为“中毒”状态时，也会在对应的影子内存区写一个特殊值，该值称为“影子值”。ASAN将8字节的主应用区内存映射为1字节的影子区内存。针对任何8字节对齐的主应用区内存，总共有9种不同的影子内存值，用于表示内存的分配状态和访问权限。

![image](/images/2024-11-27/shadow_memory.jpeg#pic_center)

- 4字中的全部8字节都未“中毒”(可访问的)，影子值是0。
- 4字中的全部8字节都“中毒”(不可访问的)，影子值是负数。
- 前k个字节未“中毒”，后8-k字节“中毒”，影子值是k。这一功能的达成是由malloc函数总是返回8字节对齐的内存块来保证的，唯一能出现该情况的场景就在申请内存区域的尾部。例如，我们申请13个字节，即malloc(13)，这样我们会得到一个完整的未“中毒”的4字和前5个字节未“中毒”、后3个字节“中毒”的4字节。

为了实现内存错误检测，ASan工具会为应用数据额外分配一块对应的shadow memory来存储相关的元数据，方法是直接将实际地址进行缩放+偏移映射到一个shadow地址，从而将整个的应用程序地址空间映射到一个shadow地址空间。

由于malloc函数返回的地址通常都至少是8字节对齐的，因此，对于应用程序堆内存上任意已对齐的8字节序列来说，它只有9种状态：前k(0=<k<=8)个字节是可寻址的，剩余8-k个不是，这样状态就可以通过一字节的Shadow Memory进行表示。ASan会将虚拟地址空间的1/8作为Shadow Memory，通过对实际地址进行缩放+offset直接将它们转换到对应的Shaodw地址。假设应用程序内存地址为Addr，那么其对应的Shadow地址为(Addr>>3)+Offset(对于1/2/4字节访问方式，原理类似)。

#### 2.2 插桩检测

概念：插桩检测是指在程序编译过程中，向源代码中插入额外的代码或检查点，以便在程序运行时收集信息或执行特定的操作。
实现方式：ASan在编译时，通过编译器插桩模块修改原有的程序代码，对每一次内存访问操作进行检查。具体来说，编译器会在每次内存读写操作前插入代码，检查目标地址的影子内存状态。如果目标地址处于“中毒”状态（即已被释放或未分配），则触发错误报告。
我们对一个内存地址的“访问”无外乎两种操作：“读”和“写”，也就是

```C
... = *address; \\ 读
*address = ...; \\ 写
```

ASan的工作依赖编译器运行时库，当开启 Address Sanitizer 之后， 运行时库将会替换掉 malloc 和 free 函数，在 malloc 分配的内存区域前后设置“投毒”(poisoned)区域, 使用 free 释放之后的内存也会被隔离并投毒，poisoned 区域也被称为 redzone。这样对内存的访问，编译器会在编译期自动在所有内存访问之前做一下 check 是否被“投毒”。所以以上的代码，就会被编译器改成这样：

![image](/images/2024-11-27/instru.png)
图：插桩前的代码和插桩后的代码对比

这样的话，当我们不小心访问越界，访问到 poisoned 的内存（redzone），就会命中陷阱，在运行时 crash 掉，并给出有帮助的内存位置的信息，以及出问题的代码位置，方便开发者排查和解决。

#### 2.3 红区投毒
概念：红区投毒是指在已分配内存的前后（称为“红区”）以及释放的内存区域中填充特殊值（即“毒药”），以便在程序运行时检测内存访问错误。
实现方式：ASAN在接管malloc和free函数后，会在已分配内存的前后创建带毒的红区，并将释放的内存区域标记为“中毒”状态。当程序尝试访问这些区域时，ASAN能够检测到并报告错误。

![image](/images/2024-11-27/redzone.png)

下图分别展示了使用未使用ASan和使用了ASan后进程内存的情况。

![image](/images/2024-11-27/redzone1.png)
图：未使用ASan的进程内存情况

![image](/images/2024-11-27/redzone2.png)
图：使用了ASan后的进程内存情况

**ASan存在的一些问题挑战**：

**问题挑战一**：存在错误漏报

对内存溢出检查：当指针的访问区域超过了redzone，对Buffer-Overflow会漏报[2]。
![image](/images/2024-11-27/fp1.png)
例如部分未对齐的越界访问
![image](/images/2024-11-27/fp2.png)
在例如越界访问落在其它已分配的内存区域上（越过红区）
![image](/images/2024-11-27/fp3.png)
释放后访问检查：目前是对该内存进行隔离，并对影子内存标记为0xFD，但这个隔离不可能永久；一但被重新复用后，也可能造成严重内存问题，有类像内存池复用崩溃问题[2]。
![image](/images/2024-11-27/fp4.png)
例如大量内存被分配又释放的情况：
![image](/images/2024-11-27/fp5.png)

**问题挑战二**：性能开销大

从基本工作原理来看，我们可以获知，打开AddressSanitizer 会增加内存占用，且因为所有的内存访问之前都进行了插桩检查是否访问了“投毒”区域的内存，此处会有额外的运行开销，对运行性能造成一定的影响，因此ASan通常只在 Debug 模式或测试场景下使用。研究表明，ASan的CPU开销约为 2 倍，代码大小开销在 50% 到 2 倍之间，内存开销很大，约为 2 倍[3]。


## 三、ASan性能优化前沿技术归纳与总结

尽管ASan的性能开销很大，但对比其它的动态内存缺陷检测器来说，其开销相当较小，如下图所示。但为了使其能更好地用到生产环境和系统级测试中，研究如何降低ASan的性能开销有着重要的意义和实际价值。

![image](/images/2024-11-27/asan_compare.png)
图：ASan与其它动态内存缺陷检测工具的对比

地址消毒器技术的性能优化领域，近年来涌现出多项前沿技术，这些技术旨在解决Asan带来的性能开销问题，大部分的研究致力于降低ASan的CPU开销，也有研究聚焦于降低ASan的内存开销。以下为近些年来致力于降低ASan性能开销的一些代表性研究工作：

![image](/images/2024-11-27/tab_overview.png)


当前最前沿的技术主要集中在消除冗余检查、设计优化的元数据结构和解耦以减少Sanitizer性能开销等。在此，我们详细介绍这三种技术:
**（1）通过消除冗余检查优化性能开销**
冗余检查是Asan性能开销的一个重要来源。为了优化性能，研究者们致力于识别和消除这些不必要的检查。一种常见的方法是使用静态分析和动态追踪技术来确定哪些内存访问是安全的，从而在运行时跳过这些访问的检查。此外，还有技术通过合并相邻的内存检查操作，或者利用程序的控制流和数据流信息来预测和避免冗余检查，从而显著减少CPU和内存的开销。这方面的代表性工作有ASAP[4]、SanRazor[8]、ASAN--[3]等。

**（2）通过设计元数据结构优化性能开销**
元数据结构在Asan中扮演着关键角色，它用于跟踪内存的状态并检测潜在的错误。然而，传统的元数据结构可能引入显著的性能开销。为了优化这一点，研究者们正在探索新型、高效的元数据结构。这些结构通过减少内存占用、提高缓存利用率和简化查找操作来降低开销。例如，使用压缩技术来减小元数据的大小，或者采用分级的数据结构来加速查找过程，都是当前研究的热点。这方面的代表性工作如GiantSan[9]和FuZZan[7]。

**（3）通过解耦减少Sanitizer性能开销**
Sanitizer的性能开销往往与其紧密集成在目标程序中的方式有关。为了降低这种开销，解耦成为一种有效的策略。通过将与Sanitizer相关的功能从主程序逻辑中分离出来，可以使其以更轻量级、更灵活的方式运行。这包括将内存检查逻辑与程序的主要执行路径分离，或者使用独立的线程或进程来执行Sanitizer的任务。解耦不仅可以减少Sanitizer对程序性能的直接影响，还有助于提高错误检测的准确性和可靠性。这方面的代表性工作如和FuZZan[7]和SAND[10]。
综上所述，当前最前沿的技术在优化Asan性能开销方面采取了多种策略，包括消除冗余检查、设计优化的元数据结构和解耦Sanitizer的功能。这些技术为降低Asan的运行时开销提供了有力的支持，并为进一步的研究和实践奠定了基础。本项目将深入研究这些技术，并通过对比评估和实践应用，探索进一步降低Asan运行时性能开销的有效方式。

### 3.1 ASAP：通过贪心消除高开销插桩满足性能需求（S&P’15）

由于引入的安全性检查会减慢程序速度，并完全消除低级语言带来的性能提升。现有的ASan优化方案更多是考虑检测性能的提升，没有考虑到帮助开发者在软件性能和安全性之间找到一个平衡点供用户选择。从用户角度出发，目标程序在某些场景下并不需要最大化检查性能，仅仅需要牺牲较小的开销满足一定的内存安全检查。

![image](/images/2024-11-27/asap1.png)

程序检测的大部分开销仅归因于少数“热”检查，而对安全性最有用的检查通常是“冷”且廉价的。基于这个观测，该工作提出了一种通过通过贪心消除高开销插桩的方法，牺牲了一定的内存安全“热”检查，从而有效降低了ASAN运行时的CPU开销。具体来说，该方法分为三步：（1）插桩：ASAP通过编译器将安全性检查插入到程序中，生成一个受保护但运行缓慢的二进制文件，用于在运行时验证程序的安全属性。（2）性能分析阶段的目的是测量每个安全性检查对程序性能的影响，即计算每个检查的成本。ASAP 使用静态指令成本模型和性能计数器收集的数据来估算每个检查的CPU周期开销。（3）根据每个检查的成本和用户指定的开销预算，ASAP 使用贪心算法选择保留尽可能多的检查，同时确保总开销不超过预算。
实验结果表明，通过在 Phoronix 和 SPEC 基准测试程序上的评估，ASAP 能够在安全性和性能之间精确选择最佳点。此外，分析了 RIPE、OpenSSL 和Python 解释器中现有的漏洞和安全漏洞，ASAP 方法提供的保护水平足以抵御所有这些漏洞。

![image](/images/2024-11-27/asap2.png)
![image](/images/2024-11-27/asap3.png)


### 3.2 FuZZan：用于模糊测试的高效ASan元数据设计（USENIX SEC’20）

Fuzzing 和 Asan 结合是排查程序缺陷的一个主流方式，在实际的 Fuzzing 过程中，内存管理是运行过程中主要来源之一。特别是大量短时间执行的程序，会有频繁的内存隐射和页表回收，大大拖慢了运行时间，最高可达6.59倍的开销。

![image](/images/2024-11-27/fuzazan1.png)

由于内存管理是运行过程的主要开销之一，因此怎么通过设计轻量的元数据减少内存管理，进一步减少运行时开销是这个工作的主要研究内容。此外对于不同的应用场景，怎么进行元数据结构的选择和维护，保证元数据结构维护信息的正确性和访问效率也是亟待解决的问题。为此，FuZZan设计了用于模糊测试的动态调整元数据结构：RB-tree 和 Min-shadow memory。
基于红黑树的元数据结构：
- RB-tree节点存储检测对象的红区信息，分配内存空间的时候创建节点，回收内存空间的时候删除节点，并在程序访问内存的时候进行判断。
- RB-tree提供了较低的启动和结束成本，但每次访问的开销相对较高，因此这种元数据结构适用于元数据访问次数较少的短期执行场景。

![image](/images/2024-11-27/fuzazan2.png)

Min-shadow memory：
- Min-shadow memory通过限制可访问的虚拟地址空间的大小，减少了所需的影子内存，这意味着只为实际使用的内存分配影子内存，而不是为整个虚拟地址空间分配。
- FuZZan可以根据程序的运行时行为动态调整 Min-shadow memory的配置，适用于具有中高元数据访问次数和执行时间较长的执行。

![image](/images/2024-11-27/fuzazan3.png)

动态元数据结构：
- FuZZan使用动态采样的方式，记录内存空间访问、删除、添加的数量，以及内存的使用数量以决定使用哪种元数据结构。
-定期的、有条件的进行采样，以适应当前输入对程序行为的影响，从而在不同的执行阶段提供最佳性能。

![image](/images/2024-11-27/fuzazan4.png)

实验评估了FuzzAn的执行效率、漏洞检测能力、元数据结构选择三个方面。
在执行效率方面，Google fuzzer 测试套件中的多个应用程序具有超过 1000 次元数据访问，因此 RB 树的平均速度总体上比 ASan 慢，但是其他策略都比原始Asan速度快。

![image](/images/2024-11-27/fuzazan5.png)

对Google fuzzer测试套件的 500,000 次完整执行的汇总，FuZZan检测性能表现更好，也表明了性能差异的主要原因是FuZZan有着更小的页错误。

![image](/images/2024-11-27/fuzazan6.png)

在漏洞检测能力方面，该研究基于Juliet测试套件针对内存损坏CWE的三种不同元数据结构模式的检测能力，FuZZan和ASan的结果相同，同时测评在24小时实验下FuZZan和ASan发现的唯一路径数量，FuZZan发现的唯一路径更多。

![image](/images/2024-11-27/fuzazan7.png)

在元数据结构选择方面，该工作评估了 Google 的模糊测试套件中部分测试程序的元数据结构切换的频率和每次元数据结构选择的频率，结果表明动态元数据结构切换是自动选择最符合程序特征的元数据结构。

![image](/images/2024-11-27/fuzazan8.png)


### 3.3 SanRazor：结合代码覆盖和数据依赖关系减少冗余判断（USENIX’21）

Asan检测过程中，相同的安全属性被反复检查会导致浪费计算资源。另一方面，在实践中，插桩后的程序运行时开销过高阻碍了它们在软件部署场景中的使用。然而，现有的通过减少插桩从而减少内存检查的方法要么使用特定的静态分析来仅删除语义上冗余的检查，要么使用启发式方法来删除昂贵的检查，在判断冗余检查的准确性上仍然不足。因此，研究如何设计正确而且高效的的冗余判断方案判别出冗余的插桩和进一步清除有着重要的意义。
SanRazor同时捕获检查的动态代码覆盖率和静态数据依赖关系，并使用提取的信息执行冗余检查分析，下图为SanRazor的主要工作流程。

![image](/images/2024-11-27/sanrazor1.png)

具体来说，SanRazor主要包括三个关键步骤：动态代码覆盖模式收集、静态数据依赖模式收集，以及消除冗余检查。

（1）动态代码覆盖模式收集
通过检测 LLVM IR 并在 br 语句之前插入计数器语句来捕获检查的动态代码覆盖率，构成检检查的动态特征。

![image](/images/2024-11-27/sanrazor2.png)

（2）静态数据依赖模式收集
在静态分析阶段，SanRazor分析从br指令开始，追溯到影响它的icmp指令的操作数并提取数据流信息并且构建依赖树，这些信息构成了检查的静态特征，有三种方案进行表示：
L0方案：收集依赖树中所有叶节点到一个集合中，这个集合包含了所有参与比较操作的变量和常量。保存是集合包括所有叶节点，这个集合代表了检查的静态特征。
L1方案：在 L0 的基础上，对集合进行规范化，移除所有常量，但保留比较语句中与每个 sanitizer 或用户检查相关的常量操作数，这样做是为了保留可能区分不同检查的关键信息。保存的集合包含了变量、全局变量、函数参数以及与比较操作相关的常量
L2方案：进一步规范化集合，移除所有常量，包括那些在比较语句中的常量，这使得方案更加激进，因为它假设所有通过指针算术表达式访问的检查都是等价的。保存的集合仅包含变量、全局变量和函数参数。

（3）消除冗余检查
- 动态模式相同：动态覆盖模式由一个三元组表示，例如 sci 的模式<sbi, stbi, sfbi>，两个检测满足 (sbi = sb j) ∧ ((stbi = stbj) ∨ (stbi = sfbj)) 则相同。
- 静态模式相同：如果两个检查 ci 和 cj 的静态模式集合 Si 和 Sj 相同，则认为它们有相同的静态模式。
- 移除动态模式和静态模式相同的检查：通过将其控制流语句的条件设置为假，使得警告/终止分支永远不会执行。

在 CPU SPEC 2006 基准上的实验结果表明，SANRAZOR 可以显著降低排错剂的开销，ASan的开销从 73.8% 降低到 28.0%-62.0%，UBSan的开销从 160.1% 降低到 36.6%-124.4% 。


### 3.4 ASan--：通过静态分析可靠地消除冗余插桩（USENIX SEC’22）

研究表明，ASan在编译阶段对程序进行了大量无用的插桩，导致引入了较高的运行时开销，限制了它在很多场景下的应用。ASan运行时对插桩的检测开销约占了运行时总开销的80.8%。因此，减少插桩可以有效减少ASAN的运行时开销，一方面需要考虑如何高效地判断出冗余插桩并移除检查，另一方面也要考虑如何保证移除冗余插桩后不损害ASan的功能性、可扩展性和可用性。
ASan--方法首次提出并实现了利用程序静态分析方法可靠地消除冗余插桩，该方法开发了四种静态优化技术，包括：（1）去除不满足条件的检查：移除那些在任何执行路径上都不会越界的检查。（2）去除重复的检查：识别并移除那些已经被其他检查覆盖的冗余检查。（3）合并邻近的检查：将空间上相邻的内存访问的检查合并为一个。（4）优化循环中的检查：将循环中不变的内存访问检查移出循环，以及合并循环中单调递增或递减的内存访问检查。

去除不满足条件的检查（Removing Unsatisfiable Checks）
- 目的：移除那些在任何情况下都不会导致越界的检查，因为这些检查是不必要的。
- 方法：通过控制流遍历和基本常量传播，识别出可以证明是界限内的堆或全局变量访问，然后移除它们的 sanitizer 检查。
- 分析：这项技术利用了编译时的分析来确保某些访问在运行时总是安全的，从而避免了运行时的检查开销。

![image](/images/2024-11-27/asan--1.png)
![image](/images/2024-11-27/asan--2.png)

去除重复的检查（Removing Recurring Checks）
- 目的：消除对同一内存位置的冗余检查，特别是当一个检查的结果可以保证后续相同位置访问的安全性时。
- 方法：使用支配分析（domination analysis）和流不敏感别名分析（flow-insensitive alias analysis）来识别并移除被其他检查所覆盖的冗余检查。
- 分析：这项技术通过分析程序的控制流图来确定哪些检查是多余的，从而减少运行时的检查次数。
- 
![image](/images/2024-11-27/asan--3.png)

合并邻近的检查（Merging Neighbor Checks）
- 目的：对于在内存中相邻的访问，将它们的 sanitizer 检查合并为一个，以减少内存访问次数。
- 方法：识别在内存中相邻的访问，并将它们的检查合并。这包括将多个检查合并为对影子内存（shadow memory）的一个检查。
- 分析：这项技术通过减少对影子内存的访问次数来降低运行时的检测开销，同时保持对内存错误的检测。
- 
![image](/images/2024-11-27/asan--4.png)

优化循环中的检查（Optimizing Checks in Loops）
- 目的：循环中的检查通常会导致开销累积，因此需要特别优化。
- 方法：包括两种优化：(1) 不变检查的重定位（Relocating Invariant Checks）：将循环中不变的内存访问检查移出循环，因为这些检查可以在循环之外执行一次。(2) 单调检查的分组（Grouping Monotonic Checks）：对于循环中单调递增或递减的内存访问，将连续迭代中的检查合并为一个。
- 分析：通过减少循环内部的检查次数，可以显著降低循环的开销，同时通过在循环外部进行一次性检查来保持检测能力。
- 
![image](/images/2024-11-27/asan--5.png)

该工作也对ASan--进行了系统性的实验评估。在功能性测试上，对 Juliet 测试套件和 Linux Flaw Project 中的内存错误进行测试，ASan-- 和 ASan 都取得了相同的结果。

![image](/images/2024-11-27/asan--6.png)
![image](/images/2024-11-27/asan--7.png)

在可扩展性评估的实验上，ASan-- 可以成功编译和构建 CPU2006 和 Chromium 并通过所有基准测试。

![image](/images/2024-11-27/asan--8.png)

在可用性评估的实验上，平均而言，ASan-- 可以将二进制文件的大小缩减 20.4%

![image](/images/2024-11-27/asan--9.png)

在性能优化评估实验方面，平均而言，ASan-- 将 ASan 的开销从 107.8% 降低到 63.3%，降低率为 41.7%。仅考虑 ASan 检查开销，降低率为 51.6%。

![image](/images/2024-11-27/asan--10.png)

此外，由于ASAN工具也常常在模糊测试中被使用，作者也在模糊测试上进行了评估。ASan--和 FuZZan 结合使用将 ASan 在 AFL 种子和 FuZZan 种子上的执行速度分别提高了 59.4% 和 59.3%，导致分支覆盖率分别提高了 8.99% 和 9.73%。 

![image](/images/2024-11-27/asan--11.png)

### 3.5 GiantSan：通过段折叠进行有效的内存检查（ASPLOS’24）

许多Sanitizer基于位置的方法面临保护密度低的问题，即每个元数据保护的字节数有限（如Asan 1:8），导致运行时开销大。
要减小运行时开销，问题的关键在于如何设计合理的元数据结构，提高元数据保护的字节数从而减少检查开销。同时，由于现有基于位置的方法无法有效处理操作级保护，怎么设计合理的方案兼容检查操作级保护和指令级保护来优化区间检查，减少查询次数，也是研究的主要难点。
GiantSan 通过段折叠进行有效的内存检查。所谓折叠段就是通过二进制折叠策略创建折叠段，这些折叠段是对连续的不含非可访问字节的段的总结。

![image](/images/2024-11-27/giantsan1.png)
![image](/images/2024-11-27/giantsan2.png)

进一步，通过检查段的折叠度来快速确定一个内存区域是否可以安全访问，分为快速检查和慢速检查。

![image](/images/2024-11-27/giantsan3.png)

此外，另一个优化就是使用历史缓存来减少对同一指针的重复元数据加载，并且保证缓存的区间尽量大。

![image](/images/2024-11-27/giantsan4.png)

实验结果表明，GiantSan通过SPEC CPU 2017基准测试展示了其性能，在运行时开销上分别比 ASan 和 ASan-- 低59.10%和38.52%。


### 3.6 SAND: 通过解耦模糊测试和Sanitizer检测减少开销（arXiv）

模糊测试结合Sanitizer（ASan、MSan、TSan） 是排查软件缺陷常用的高效解决方案，但是这两者结合往往采用同步执行的方式，会有大量的Sanitizer检测。目前研究表明模糊测试中的大多数种子不会引发错误，只有 1.3% 的输入是触发 bug 的。而且，在常见的模糊测试结合Sanitizer的测试方案中，多种类型的Sanitizer会互斥无法同时执行。该工作解决的主要问题是“怎么实现有选择性地对部分种子进行Sanitizer检查，应该选择怎么样的种子”。
SAND的核心思想是将 Fuzzing 和 Sanitizer 解耦，仅对 Fuzzing 过程来说是有趣的种子进行 Sanitizer，以此减少检测。SAND的主要流程如下图所示。

![image](/images/2024-11-27/sand1.png)

（1）执行模式获取：采用执行模式近似表示程序执行的唯一路径，可以通过模糊测试工具获取覆盖率过程中设计的 bitmap 获取到路径模式的位图信息。

![image](/images/2024-11-27/sand2.png)

（2）解耦Fuzzing和Sanitizer：解耦模糊测试程序和Sanitizer检测程序，生成多个二进制版本，对于模糊测试有趣并且具有唯一执行路径的种子进行Sanitizer检测。

![image](/images/2024-11-27/sand3.png)

实验结果表明，在24小时测试内，与启用ASan/UBSan和MSan的程序进行模糊测试相比，SAND分别实现了 2.6 倍和 15 倍的吞吐量，检测到的错误增加了 51% 和 242%。


## 四、实证研究

### 4.1 实验设计

#### 4.1.1 研究问题
本实验的设计旨在回答以下研究问题
- 问题一：ASan带来的运行时开销具体是在哪些地方产生
- 问题二：当前基于ASan进行性能优化的State-of-the-art工具在**CPU开销**的优化方面能达到何种程度？
- 问题三：当前基于ASan进行性能优化的State-of-the-art工具在**内存开销**的优化方面能达到何种程度？
- 问题四：当前基于ASan进行性能优化的State-of-the-art工具在**漏洞检测能力**方面的表现如何？
- 问题五：当前基于ASan进行性能优化的State-of-the-art工具在**代码大小开销**方面的表现如何？

#### 4.1.2 对比的工具
- ASan
- ASan--
- GiantSan
- SanRazor
- LFP

#### 4.1.3 数据集
- **CPU SPEC 2006**：一个行业标准化的、CPU密集型的基准测试套件，由12个整数程序和7个浮点程序组成。
- **23个含CVE漏洞的真实程序**：这组程序包含已知的安全漏洞，用于测试工具在真实场景下检测内存安全错误的能力。

#### 4.1.4 评价指标
- **CPU开销**: 这是指工具在运行时对CPU资源或执行时间的额外消耗。一个高效的工具应该尽可能地减少这种开销，以提高整体性能。
- **内存开销**: 这指的是工具在运行时占用的额外内存空间。内存开销越小，工具对系统资源的影响就越小。
- **内存安全错误的检出率**: 这是评估工具有效性的关键指标，表示工具能够正确检测出内存安全错误的比例。检出率越高，工具的检测能力越强。
- **代码大小**: 这指的是工具引入的额外代码量或其对原始程序代码大小的增加。代码大小开销越小，说明工具对程序体积的影响越小。

#### 4.1.5 实验环境
- CPU: Intel Xeon Platinum 8255C
- Operating System: Ubuntu 18.04
- CPU MHz: 2494.140
- Memory: 62 GB
- Compiler: clang/clang++

### 4.2 研究问题一：ASan带来的运行时开销具体是在哪些地方产生的

我们使用ASan的默认设置对SPEC CPU2006中的程序进行测试，以测量其带来的总体运行时开销。为了确定ASan的运行时开销具体是在哪些地方产生，我们对ASan做了以下修改，并分别评估它们的性能开销：
- 检查器检查（Sanitizer_Chk）：禁用对内存访问（包括loads和store指令）的检查器检查。
- 拦截器（Interceptor）：禁用对C标准库（如memcpy、memset和memmove等）的拦截和替换。
- 投毒（Poisoning）：跳过红区的毒化过程。
- 日志记录（Logging）：删除中间日志记录，特别是malloc和free期间的堆栈跟踪日志。
- 堆管理（Heap Management）：将ASan的堆管理撤销为标准C库的堆管理。
- 红区（Redzone）：删除剩余的红区和相关操作。

![image](/images/2024-11-27/rq1.png)

### 4.3 研究问题二：基于ASan进行性能优化的SOTA工具的CPU开销评估

运行时间开销（秒），Ratio表示各个工具跟未开启ASan检测的原始程序性能开销的比值，实验根据各个工具在SPEC CPU2006基准测试套件上17个测试程序的运行时间开销进行统计对比，表明现有ASan优化工具的平均性能开销在原始程序开销的150%以上。

![image](/images/2024-11-27/rq2.png)

### 4.4 研究问题三：基于ASan进行性能优化的SOTA工具的内存开销评估

运行内存开销（MB），Ratio表示各个工具跟未开启ASan检测的原始程序内存开销的比值，实验根据各个工具在SPEC CPU2006基准测试套件上17个测试程序的运行内存开销进行统计对比，表明现有ASan优化工具的平均内存开销在原始程序开销的220%以上，与原始ASan相比差值很小。值得注意的是，对于未开启ASan检测的原始程序占用虚拟内存空间为417MB，而使用ASan工具进行检测的程序因为使用影子内存导致占用的虚拟内存空间都高达20T。

![image](/images/2024-11-27/rq3.png)

### 4.5 研究问题四：基于ASan进行性能优化的SOTA工具的漏洞检测能力评估

各个工具在23个含CVE漏洞的真实程序进行测试，评估各个工具的漏洞检测能力，实验表明大部分ASan优化工具具备发现CVE漏洞的能力。

![image](/images/2024-11-27/rq4.png)

### 4.5 研究问题五：基于ASan进行性能优化的SOTA工具的代码大小评估

二进制文件大小（KB），Ratio表示各个工具跟未开启ASan检测的原始程序二进制文件大小的比值，实验根据各个工具在SPEC CPU2006基准测试套件上17个测试程序的二进制文件的大小进行统计对比，表明现有ASan优化工具的平均二进制文件大小在原始程序二进制文件大小的170%以上。

![image](/images/2024-11-27/rq5.png)


## 五、总结与下阶段工作：

### 5.1 总结

- 当前面向ASan性能开销优化的主要方法包括（1）通过程序动静态分析技术消除冗余检查；（2）通过设计优化的元数据结构即可以降低ASan的CPU开销；（3）结合模糊测试与ASan解耦，降低Asan的检测次数。其中方法（1）与方法（2）是互补的，有一定可行性。方法（3）的通用性和可扩展性较低，不适用于本项目。
- 实验表明，原始ASan的CPU消耗在2倍左右，内存消耗在2.5倍以上，额外代码大小开销在50%到4倍之间。目前的前沿方法已经能实现将CPU消耗降低到1.5倍左右，但内存消耗相比始ASan相比未见明显改善，部分工具的实现略微牺牲了漏洞检测能力。

### 5.2 下阶段工作：

- 基于目前的调研结果，短期可尝试方法（1）和方法（2）的结合，并进行工具的实现。
- 鉴于当前的SOTA方法在ASan的内存开销优化方面未见明显改善，后续研究应当注重ASan在内存消耗方面的优化。
- 进一步，应当研究和提出其它创新性的方法来改进ASan的性能开销。


## Appendix

- [论文分享 | AddressSanitizer: 一个快速的内存地址错误检查器](https://zhuanlan.zhihu.com/p/697195679)
- [AddressSanitizer算法及源码解析](https://blog.csdn.net/juS3Ve/article/details/80879159)
- [ASAN Pass【源码分析】（一）——简单分析](https://blog.csdn.net/clh14281055/article/details/119276042)
- [ASAN Pass【源码分析】（二）——调试环境准备](https://blog.csdn.net/clh14281055/category_11176781.html?spm=1001.2014.3001.5482)
- [ASAN Pass【源码分析】（三）——初始化](https://blog.csdn.net/clh14281055/article/details/119465513)
- [ASAN Pass【源码分析】（四）——运行](https://blog.csdn.net/clh14281055/article/details/119514551)
- [ASAN Pass【源码分析】（五）——插桩](https://blog.csdn.net/clh14281055/article/details/119523477)
- [ASAN Pass【源码分析】（六）——全局变量插桩](https://blog.csdn.net/clh14281055/article/details/122896319)


## 参考文献

[1]Serebryany K, Bruening D, Potapenko A, et al. AddressSanitizer: A fast address sanity checker[C]//2012 USENIX annual technical conference (USENIX ATC 12). 2012: 309-318.
[2]Han W, Joe B, Lee B, et al. Enhancing memory error detection for large-scale applications and fuzz testing[3]//Network and Distributed Systems Security (NDSS) Symposium 2018. 2018.
[4]Zhang Y, Pang C, Portokalidis G, et al. Debloating address sanitizer[C]//31st USENIX Security Symposium (USENIX Security 22). 2022: 4345-4363.
[5]Wagner J, Kuznetsov V, Candea G, et al. High system-code security with low overhead[C]//2015 IEEE Symposium on Security and Privacy. IEEE, 2015: 866-879.
[6]Han W, Joe B, Lee B, et al. Enhancing memory error detection for large-scale applications and fuzz testing[7]//Network and Distributed Systems Security (NDSS) Symposium 2018. 2018.
[8]Hardware-assisted AddressSanitizer Design Documentation Hardware-assisted AddressSanitizer Design Documentation — Clang 20.0.0git documentation (https://clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html)
[9]Jeon Y, Han W H, Burow N, et al. FuZZan: Efficient sanitizer metadata design for fuzzing[C]//2020 USENIX Annual Technical Conference (USENIX ATC 20). 2020: 249-263.
[10]Zhang J, Wang S, Rigger M, et al. SANRAZOR: Reducing redundant sanitizer checks in C/C++ programs[11]//15th USENIX Symposium on Operating Systems Design and Implementation (OSDI 21). 2021: 479-494.
[12]Ling H, Huang H, Wang C, et al. GIANTSAN: Efficient Memory Sanitization with Segment Folding[C]//Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 2024: 433-449.
[13]Kong Z, Li S, Huang H, et al. SAND: Decoupling Sanitization from Fuzzing for Low Overhead[J]. arXiv preprint arXiv:2402.16497, 2024.