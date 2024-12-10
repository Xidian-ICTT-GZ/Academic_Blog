# GiantSan: Efficient Memory Sanitization with Segment Folding



> 通过段折叠实现高效内存清理
>
> 作者：Hao Ling, Heqing Huang, Chengpeng Wang, Yuandao Cai, Charles Zhang
>
> 单位：The Hong Kong University, City University of Hong Kong
>
> 期刊：ASPLOS ’24
>
> 论文链接：[GiantSan: Efficient Memory Sanitization with Segment Folding](https://5hadowblad3.github.io/files/asplos24-GiantSan.pdf)

| Recorder | Date      | Categories                |
| -------- | --------- | ------------------------- |
| 林志伟   | 2024-12-9 | Testing, Dynamic Analysis |



## 1. Issue

- **内存安全问题突出**：C 和 C++ 等语言因指针操作致内存安全违规频发，如越界读写、使用释放后内存等，在 2022 CWE 软件弱点排名居前，凸显内存清理技术关键地位。

- 现有方法局限显著：
  - **指针式方法依赖强**：靠跟踪指针安全区域及传播标签保障内存安全，依赖指针类型信息，遇指针整数转换或无类型信息库时，标签传播失败致内存保护失效，且需大量指令处理指针算术与标签传播，增加运行开销。
  - **位置式方法效率低**：依内存字节地址可达性记录建模，兼容性高但保护密度不足。因区分字节状态至少需 1 位，多按 8 字节段存储段状态，访问内存常大量加载解码影子字节，如 AddressSanitizer 检查 1KB 区域需加载 128 段状态，致高运行时开销。



## 2. Background

**内存检测办法可以分为基于指针的检查和基于位置的检查：**

- 基于指针：
  - 基于指针的方法从指针的角度对内存进行建模，跟踪每个指针可以安全访问的内存区，将指针和标记封装在一个新的指针表示中，并使用标记作为安全区域的绑定或检索绑定的索引。
  - 基于指针的方法通过检查正在访问的内存区域是否在安全范围内来保护内存操作，检测次数比较少，但是兼容性比较差。
- 基于位置：
  - 基于位置的方法通过记录哪些字节是可寻址的，从内存字节的角度对内存进行建模，字节状态记录在紧凑的影子存储器中，基于位置的方法检查影子存储器以检查每个访问字节的状态。
  - 基于位置的方法没有这样的限制，它们必须将操作分解为指令并单独检查每条指令，以确保没有不可寻址的字节被访问，但是开销也会比较大。



**操作级保护和指令级保护：**

- 操作级保护的目的是保护一个由多个指令组成的内存操作作为一个整体，指令级保护对每个指令分别进行保护。
- 操作级保护需要的检查比指令级保护少得多，因此检测开销比较小，此外操作级保护还可以通过缓存减少元数据加载。
- **现有的基于位置的方法在指令级保护方面存在对大内存区域访问的判断低效和历史缓存低效的不足之处，都是由于保护密度低造成的。**



## 3. introduction

- 内存安全检查器使用运行时元数据来建模内存并帮助发现程序中的内存错误，基于位置的方法应用比较广泛，但是基于位置的方法面临保护密度低的问题，即每个元数据保护的字节数有限，导致运行时开销大。
- 观察分析：
  - 运行时内存检查和元数据加载是主要开销；
  - 根据内存对齐情况， 一些连续的字节必须同时是可寻址的或不可寻址的；
  - 程序中大部分对内存访问都是安全的，也就是可寻址的，内存错误只是小部分；
- 本文提出了一种带有段折叠的新影子编码，这种方法使用更少的元数据来保护更大的内存区域，加快了内存 sanitization 的速度。
- 实现了一个名为 GiantSan 的工具，并通过SPEC CPU 2017基准测试展示了其性能。与现有的最先进方法相比，GiantSan 在运行时开销上分别比 ASan 和 ASan-- 低59.10%和38.52%。



## 4. Design

- 段折叠算法
  - **折叠策略设计**：采用递归二进制折叠，将连续可达字节的段（2^x个）组合为新折叠段，以折叠度 x 结合 8 字节对齐，所有段状态与 x 存于 8 位整数，大幅提升保护密度，减少元数据加载。
  - **影子编码定义**：新编码含折叠段、部分可达段及不可达段状态编码，通过比较影子字节值与简单不等式判断段折叠度及内存区域安全性，更新影子内存计算复杂度与传统方法持平。
- 区域检查优化
  - **检查流程高效化**：查内存区域 [L, R) 时，先快速检查首段折叠段确定的安全区域 [L, L + u)，多数情况可判定；否则慢速检查区域前缀、后缀及末段是否含不可达字节，利用折叠段特性，能在 O(1) 时间处理任意大小区域，提高检查效率。
- 历史缓存机制
  - **缓存原理及更新**：缓存指针访问的最后折叠段为临时准边界，后续访问先依此判断，在准边界内免额外元数据加载。按需跳过折叠段定位准边界，更新准边界次数上限，降低运行时检查成本。
- 操作级保护提升
  - **锚点增强准确性**：设小 redzone 并选缓冲区基指针为锚点，查锚点与访问位置间 redzone 防绕过，如访问 y[j] 以 y 为锚点查更大区域，仅用 1 字节 redzone，解决 redzone 大小权衡难题，提升内存保护精度与效率。
  - **静态分析降冗余**：编译时生成指令级检查后，经别名检查消除、循环内检查提升等静态分析手段，合并冗余检查，减少检查次数，充分发挥操作级保护处理任意区域与缓存优势，提升整体性能。

<img src="https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241024000358626.png" alt="image-20241024000358626" style="zoom: 67%;" />



### 4.1 Shadow Encoding in GiantSan

<img src="https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241024000311448.png" alt="image-20241024000311448" style="zoom:67%;" />

1. GiantSan 采用8字节对齐的段，每个段用8位数据类型存储元数据。

2. GiantSan 通过二进制折叠策略来创建“折叠段”，这些折叠段是对连续的“好”段（即不含非可访问字节的段）的总结，以减少需要加载的元数据量。

   ```
   m[p] is an 8-bit unsigned integer that can store values within [0, 256).
   
   m[p] =>  
   	64 − i, 	the p-th segment is an (i)-folded segment  
   	72 − k, 	the p-th segment is a k-partial segment  
   	> 72, 		error codes
   ```

3. 影子内存保存了不同状态码来表示段的状态，包括折叠度、部分段和错误码。

4. 对于状态码x，表示连续x个段是可以访问的，也就是 2^x 的地址空间是可以访问的。在64位操作系统中，虚拟地址空间不会超过 2^64，因此只需要6位就可以表示所有的折叠度。

5. 折叠段的编码方式允许 GiantSan 以对数时间复杂度来记录和检查内存区域的可访问性。



### 4.2 Region Checking

<img src="https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241024001519364.png" alt="image-20241024001519364" style="zoom:67%;" />

- GiantSan通过检查段的折叠度来快速确定一个内存区域是否可以安全访问，分配的对象至多有一个部分段，并且分配区域内的所有剩余段都被折叠。
- 如果内存区域 [L, R) 中除最后一个段之外的所有段都是“好”段，并且最后一个段中的第 R mod 8 个字节是可寻址的，则该内存区域是安全的。

- **快速检查（Fast Check）**：如果一个区域被一个高折叠度的段覆盖，那么这个区域就是安全的。

  ![image-20241024002925292](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241024002925292.png)

- **慢速检查（Slow Check）**：对于不能通过快速检查确定的区域，GiantSan 会判断能否被两个段（prefix and suffix）包含。

  ![image-20241024002948250](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241024002948250.png)

- 因为二进制折叠的性质，其他访问超出范围的会出现报告错误。



### 4.3 History Caching

![image-20241210161038451](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161038451.png)

使用历史缓存来减少对同一指针的重复元数据加载：

- **准边界**：GiantSan 为每个指针维护一个准边界，这个边界是基于之前访问的折叠段确定的，当新的访问发生在这个边界内时，就不需要再次加载元数据，在检查超出边界时重新生成准边界。
- **缓存逻辑**：GiantSan 在循环中使用准边界来减少元数据的加载，从而加快运行时检查的速度。



### 4.4 Check Instance Generation

![image-20241210161103371](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161103371.png)

GiantSan 通过生成操作级检查来减少运行时开销：

- **锚点增强**：GiantSan通过选择锚点来增强对内存访问的保护，这允许GiantSan在只需要一个字节的红区的情况下，精确地保护内存。
- **操作级检查**：GiantSan使用静态分析来合并和消除不必要的检查，从而减少运行时检查的数量，这包括消除别名检查和提升循环中的检查。



## 5. Elevation

- **性能评估**：SPEC CPU2017 基准测试示，GiantSan 较 ASan 和 ASan-- 运行时开销分别降 59.10% 与 38.52%，仅 5 项目慢于 LFP，整体性能优，凸显段折叠算法在新影子编码中的高效。
  
  <img src="https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161203598.png" alt="image-20241210161203598" style="zoom:80%;" />
  
  
  
- **检测能力**：Juliet Test Suite、Linux Flaw Project 及 Magma Benchmark 检测中，GiantSan 与 ASan、ASan-- 多数情况检测结果同，LFP 因分配策略多误报。
  
  ![image-20241210161318667](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161318667.png)
  
  ![image-20241210161308083](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161308083.png)
  
  ![image-20241210161332238](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161332238.png)
  
  
  
## 6. limitation

- **反向遍历负优化**：GiantSan 单向折叠导致反向遍历效率降，如 Perlbench 测试，随机与正向遍历比 ASan 快 1.48 倍与 1.07 倍，反向遍历因锚点增强指令慢 1.39 倍，因无法高效从高地址预测低地址可达性，幸反向遍历在实际程序占比少（SPEC CPU 2017 中仅 0.39%），且可通过调整检测策略缓解。
  
  ![image-20241210161623091](https://cdn.jsdelivr.net/gh/KylinLzw/MarkdownImage/img/image-20241210161623091.png)
  
  
  
- **假阴性**：同现有方法，GiantSan 难检子对象溢出，内存隔离机制有小概率绕过风险，但实践中概率低且误报少。
