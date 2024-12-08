​									*Greybox Fuzzing for Concurrency Testing*

![image](https://github.com/Xidian-ICTT-GZ/Academic_Blog/blob/main/images/2024-12-8/%E4%B9%90%E8%A7%82%E4%B8%8E%E6%82%B2%E8%A7%82%E5%B9%B6%E5%8F%91%E6%B5%8B%E8%AF%95.png)


在本文中提出一种有偏的随机搜索，利用reads-from信息来对交错分区进行划分。调度器利用reads-from对偏向新发现的分区。针对并发程序开发的并发中心灰盒模糊测试（RFF）并未枚举所有可能的交错或来自“reads-from”分区的事件，而是进行自适应的、随机化的搜索以减少分区空间。

![image]([C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207100545044.png](https://github.com/Xidian-ICTT-GZ/Academic_Blog/blob/main/images/2024-12-8/%E7%A4%BA%E4%BE%8B%E7%A8%8B%E5%BA%8F.png))

![image]([C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207165741404.png](https://github.com/Xidian-ICTT-GZ/Academic_Blog/blob/main/images/2024-12-8/reads-from%E5%AF%B9%E7%A4%BA%E4%BE%8B.png)



**有偏随机搜索**：调度器不会进行完全随机的调度，而是基于“reads-from”关系在搜索空间内进行引导。这种有偏搜索可以有效减少无关调度的尝试，集中探索潜在的错误路径。

**调度空间划分**：调度空间被划分成多个区域，每个区域对应于特定的“reads-from”关系。调度器优先探索这些分区，以便找到可能导致并发错误的交错。

![image-20241207101324126](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207101324126.png)![image-20241207104633444](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207104633444.png)

抽象调度由于是部分描述，因此通常可以找到可行的实现。此外，与具体调度相比，抽象调度可以减少存储和搜索空间的开销。

我们方法的有效性关键在于对并发程序执行的语义处理——我们识别何时两个执行是语义上等价的，并系统地避免从同一等价类中探索多个交错

为了推动执行满足抽象调度约束，我们使用优先级改变来延迟或立即执行这些约束中的事件。然而，仅仅在相关事件都已启用时提高或降低它们的优先级，通常不足以确保满足所需的抽象调度约束。相关事件通常在执行轨迹中相隔很远，因此很可能不会同时启用，除非进一步的干预。



![image-20241207105603317](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207105603317.png)



![image-20241207105620863](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207105620863.png)

POS算法即给每个事件分配了一个随机分数，并选择最高分的事件来作为下一个执行事件，并重置该事件及竞争事件的分数。因此当调度器无法明确做出调度决定的时候，就会采用POS算法来进行选择。

RFF只存储抽象调度，因为存储具体调度会导致存储开销过大。抽象调度只对执行的部分进行描述，通常可以通过调度器找到可行的实现。具体调度空间比抽象调度空间大得多，抽象调度可以更有效地避免冗余搜索。

![image-20241207101324126](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207101324126.png)

再利用用户模式调度器来执行插桩之后的程序，尝试偏向满足新的抽象调度的约束。执行结束之后再记录实际线程交错的具体调度，分析是否存在有趣行为，若有，则将此次执行的变异后的调度存入调度合集中。

![image-20241207153325277](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207153325277.png)

![image-20241207153345115](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207153345115.png)

我们的评估主要目标是回答以下四个研究问题：

1. 此工具在发现漏洞方面是否比其他先进的并发测试技术更有效？
2. 对抽象调度空间的关注如何提升了方法的有效性？
3. 在 reads-from 空间中的搜索分布有多均匀？
4. 利用 reads-from 信息的替代方法是否能和基于模糊测试启发的偏随机搜索一样有效？



![image-20241207163222880](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207163222880.png)

![image-20241207163500643](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207163500643.png)

![image-20241207171232201](C:\Users\16695\AppData\Roaming\Typora\typora-user-images\image-20241207171232201.png)
