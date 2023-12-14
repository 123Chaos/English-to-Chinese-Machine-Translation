# English to Chinese machine translation
### 试验步骤分析
- 编程环境
	- 语言：Python
	- 深度学习框架：Pytorch

- 数据预处理方面
	- 输入端进行数据预处理，将非英文字符、非数字、非,.?!的字符替换成空格
	- 输入端标点前加一个空格 方便按照空格进行单词划分
	- 限制单词数量，构建中英对照词典
	- 预训练部分语法，并且在训练的过程中继续更新

- NMT模型
	- 构建了基于RNN的seq2seq模型，并且设置hidden中间层
	- 实现了基于dot product的attention机制

- 按照BLEU设置了评估的指标

### 实验分析
- 使用jieba对中文进行分词，使用空格对英文进行分词
- 过滤单词数组长度小于某个值的数组并进行序列化
- 生成torch-tensor使用的对应张量
- 训练时使用的损失函数是NLL
- 训练使用了teacher-forcing进行函数损失的评估，每次训练时生成一个随机数，当这个数大于某个常数的时候，就调用teacher-forcing方法，加快损失函数的拟合，拟合效果较好
- 由于评估集有限以及设计了较为简易的指标，最终评估效果可能较差

### 心得体会
- 本实验难度主要体现在AttentionDecoderRNN的构建、整套系统的连接、模型的训练参数
- 其实我自己本身不常写Python，这次实验从课程角度来讲，学会了文本机器翻译的大致流程，从个人能力来讲，也体会到Python相比于JS、C++、JAVA的生态、操作难度的优越性和性能、严谨性、工程化的天然劣势。如果有其他语言的经验，学会Python语法并上手进行开发估计只需要一个小时。

### 产出
- 训练函数损失 - seq2seqLoss.png
- Attention输出 - seq2seqAttention
- srcCode - main.py
- others - 测试集、评估集、训练集