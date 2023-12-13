# Version 1.0.0
# Write by renming
# Torch on Apple M2 Pro CPU

import unicodedata
import re
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import time
import math
import matplotlib.pyplot as plt
from torch import optim

# ----------------------------- 初始化 START -----------------------------

# DATA_PATH
TEST_SRC = './test/src.jsonl'
TEST_TARGET = './test/target.jsonl'
TRAIN_SRC = './train/src.jsonl'
TRAIN_TARGET = './train/target.jsonl'
VALIDATION_SRC = './validation/src.jsonl'
VALIDATION_TARGET = './validation/target.jsonl'

# START_FLAG
SOS_TOKEN = 0

# END_FLAG
EOS_TOKEN = 1

# 限制语言对的最大长度
MAX_LENGTH = 10

# 设置带有指定前缀的数据
EN_PREFIX = ("i am ", "i m ", "he is", "he s ", "she is", "she s ", "you are", "you re ", "we are", "we re ", "they are", "they re ")

# 设置TEACHER_FORCING比率
TEACHER_FORCING_RATIO = 0.5

class Lang:
    def __init__(self, languageName):
        # eg. zh_cn
        self.languageName = languageName
        self.word2index = {"SOS": 0, "EOS": 1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        # 如果是英文
        if self.languageName == 'en':
            for word in sentence.split(' '):
                self.addWord(word)
        # 如果是中文
        elif self.languageName == 'zh_cn':
            raw_sentence = jieba.cut(sentence)
            for word in raw_sentence:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# ----------------------------- 初始化 END -----------------------------


# ----------------------------- 数据清洗 START -----------------------------

# 将unicode码转为Ascii码，可以去掉一些语言中的重音标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 规范化英文字符串
def normalizeString(s, languageName):
    if languageName == 'en':
        # 使字符变为小写并去除两侧空白符
        s = unicodeToAscii(s.lower().strip())
        # 在标点前加空格
        s = re.sub(r"([.!?])",r" \1", s)
        # 将英文、数字以外的字符替换成空格
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    if languageName == 'zh_cn':
        # 暂时不用处理
        pass
    return s

# en = Lang('en')
# en.addSentence(normalizeString("我是那种每周在男生宿舍 被打到出血的那种人 直到一个老师把我从这种生活中解救出来", 'zh_cn'))
# print("word2index:", en.word2index)
# print("index2word:", en.index2word)
# print("n_words:", en.n_words)

# ----------------------------- 数据清洗 END -----------------------------


# ----------------------------- 数据处理 START -----------------------------

# 读取jsonl文件并进行处理
def read(srcPath, targetPath):
    srcList = []
    targetList = []
    pairs = []
    srcLang = Lang('en')
    targetLang = Lang('zh_cn')

    # 读取文件
    with open(srcPath, 'r') as f:
        for line in f:
            text = json.loads(line)['text']
            srcList.append(normalizeString(text, 'en'))
    with open(targetPath, 'r') as f:
        for line in f:
            text = json.loads(line)['text']
            targetList.append(normalizeString(text, 'zh_cn'))

    # 获取字符对,截取src和target的最短长度，防止出现异常数据
    length = min(len(srcList), len(targetList))
    for i in range(0, length):
        pairs.append([srcList[i], targetList[i]])
    return srcLang, targetLang, pairs

# 处理单条语言对
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and p[0].startswith(EN_PREFIX)

# 处理多条语言对
def filterPairs(pairs):
    # 过滤长度超过限制的句子
    return [pair for pair in pairs if filterPair(pair)]

# 处理字符串映射
def prepareData(srcPath, targetPath):
    srcLang, targetLang, pairs = read(srcPath, targetPath)
    pairs = filterPairs(pairs)
    for pair in pairs:
        srcLang.addSentence(pair[0])
        targetLang.addSentence(pair[1])
    return srcLang, targetLang, pairs

# 获取单个模型输入所需要的张量
def tensorFromSentence(lang, sentence, languageName):
    if languageName == 'en':
        indexes = [lang.word2index[word] for word in sentence.split(' ')]
    elif languageName == 'zh_cn':
        indexes = [lang.word2index[word] for word in jieba.cut(sentence)]
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

# 获取pair输入所需要的张量
def tensorsFromPair(srcLang, targetLang, pair):
    return tensorFromSentence(srcLang, pair[0], 'en'), tensorFromSentence(targetLang, pair[1], 'zh_cn')

# srcLang, targetLang, pairs = prepareData(TEST_SRC, TEST_TARGET)
# fpairs = filterPairs(pairs)
# print('过滤后的前五个pair: ', fpairs[:5])
# print("srcLang_n_words:", srcLang.n_words)
# print("targetLang_n_words:", targetLang.n_words)
# print(random.choice(pairs))
# print(tensorsFromPair(srcLang, targetLang, pairs[0]))

# ----------------------------- 数据处理 END -----------------------------

# ----------------------------- 构建基于GRU的编码器 START -----------------------------

class EncoderRNN(nn.Module):
    # inputSize: 解码器的输入尺寸即源语言的词表大小
    # hiddenSize: GRU的隐层节点数, 也代表词嵌入维度, 同时又是GRU的输入尺寸
    def __init__(self, inputSize, hiddenSize):
        super(EncoderRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize)

    # input: 源语言的Embedding层输入张量
    # hidden: 编码器层gru的初始隐层张量
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)

# srcLang, targetLang, pairs = prepareData(TEST_SRC, TEST_TARGET)
# pair = pairs[0]
# pairTensor = tensorsFromPair(srcLang, targetLang, pair)
# hiddenSize = 25
# inputSize = 20
# input = pairTensor[0][0]
# hidden = torch.zeros(1, 1, hiddenSize)
# encoder = EncoderRNN(inputSize, hiddenSize)
# encoderOutput, hidden = encoder(input, hidden)
# print("EncoderRNN:", encoderOutput)

# ----------------------------- 构建基于GRU的编码器 END -----------------------------

# ----------------------------- 构建基于GRU的解码器 START -----------------------------

class DecoderRNN(nn.Module):
    # hiddenSize: 解码器中GRU的输入尺寸，也是它的隐层节点数
    # outputSize: 整个解码器的输出尺寸, 也是我们希望得到的指定尺寸即目标语言的词表大小
    def __init__(self, hiddenSize, outputSize):
        super(DecoderRNN, self).__init__()
        self.hiddenSize = hiddenSize
        # 实例化Embedding层对象, 它的参数output表示目标语言的词表大小
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        # 实例化GRU对象，它的输入尺寸和隐层节点数相同
        self.gru = nn.GRU(hiddenSize, hiddenSize)
        # 实例化线性层, 对GRU的输出做线性变化
        self.out = nn.Linear(hiddenSize, outputSize)
        # 使用softmax进行处理，以便于分类
        self.softmax = nn.LogSoftmax(dim=1)

    # input: 目标语言的Embedding层输入张量
    # hidden: 解码器GRU的初始隐层张量
    def forward(self, input, hidden):
        # 将输入张量进行embedding操作, 并使其形状变为(1,1,-1), -1代表自动计算维度
        output = self.embedding(input).view(1, 1, -1)
        # 使用relu函数对输出进行处理，根据relu函数的特性, 将使Embedding矩阵更稀疏，以防止过拟合
        output = F.relu(output)
        # 把embedding的输出以及初始化的hidden张量传入到解码器gru中
        output, hidden = self.gru(output, hidden)
        # 因为GRU输出的output是三维张量，第一维没有意义，因此通过output[0]来降维再传给线性层做变换, 最后用softmax处理以便于分类
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)
    
# srcLang, targetLang, pairs = prepareData(TEST_SRC, TEST_TARGET)
# pair = pairs[0]
# pairTensor = tensorsFromPair(srcLang, targetLang, pair)
# hiddenSize = 25
# outputSize = 10
# input = pairTensor[1][0]
# hidden = torch.zeros(1, 1, hiddenSize)
# decoder = DecoderRNN(hiddenSize, outputSize)
# output, hidden = decoder(input, hidden)
# print("DecoderRNN:", output)

# ----------------------------- 构建基于GRU的解码器 END -----------------------------

# ----------------------------- 构建基于GRU和Attention的解码器 START -----------------------------

class AttentionDecoderRNN(nn.Module):
    # hiddenSize: 解码器中GRU的输入尺寸，也是它的隐层节点数
    # outputSize: 整个解码器的输出尺寸, 目标语言的词表大小
    # dropoutP: dropout层时的置零比率，默认0.1,
    # maxLength: 句子的最大长度
    def __init__(self, hiddenSize, outputSize, dropoutP=0.1, maxLength=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.dropoutP = dropoutP
        self.maxLength = maxLength
        # 实例化Embedding层
        self.embedding = nn.Embedding(self.outputSize, self.hiddenSize)
        self.attention = nn.Linear(self.hiddenSize*2, self.maxLength)
        self.attentionCombine = nn.Linear(self.hiddenSize * 2, self.hiddenSize)
        self.dropout = nn.Dropout(self.dropoutP)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.outputSize)
    
    # input: 源数据输入张量
    # hidden: 初始的隐层张量
    # encoderOutputs: 解码器的输出张量
    def forward(self, input, hidden, encoderOutputs):
        # 根据结构计算图, 输入张量进行Embedding层并扩展维度
        embedded = self.embedding(input).view(1, 1, -1)
        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)

        # 进行attention的权重计算

        # 第一步,将Q，K进行纵轴拼接, 做一次线性变化, 最后使用softmax处理获得结果
        attentionWeights = F.softmax(self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        # 第二步, 将得到的权重矩阵与V做矩阵乘法计算, 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        attnApplied = torch.bmm(attentionWeights.unsqueeze(0), encoderOutputs.unsqueeze(0))
        
        # 第三步, 通过取[0]是用来降维, 根据第一步采用的计算方法, 需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((embedded[0], attnApplied[0]), 1)

        # 第四步, 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        output = self.attentionCombine(output).unsqueeze(0)

        # attention结构的结果使用relu激活
        output = F.relu(output)

        # 将激活后的结果作为gru的输入和hidden一起传入其中
        output, hidden = self.gru(output, hidden)

        # 最后将结果降维并使用softmax处理得到最终的结果
        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output, hidden, attentionWeights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)
    
# srcLang, targetLang, pairs = prepareData(TEST_SRC, TEST_TARGET)
# pair = pairs[0]
# pairTensor = tensorsFromPair(srcLang, targetLang, pair)
# hiddenSize = 25
# outputSize = 10
# input = pairTensor[1][0]
# hidden = torch.zeros(1, 1, hiddenSize)
# encoderOutputs = torch.randn(10, 25)
# decoder = AttentionDecoderRNN(hiddenSize, outputSize)
# output, hidden, attentionWeights = decoder(input, hidden, encoderOutputs)
# print(output)
# print(hidden)
# print(attentionWeights)

# ----------------------------- 构建基于GRU和Attention的解码器 END -----------------------------

# ----------------------------- 构建模型训练函数, 并进行训练 START -----------------------------

# inputTensor：源语言输入张量
# targetTensor：目标语言输入张量
# encoder, decoder：编码器和解码器实例化对象
# encoderOptimizer, decoderOptimizer：编码器和解码器优化方法
# criterion：损失函数计算方法
# maxLength：句子的最大长度
# 训练函数
def train(inputTensor, targetTensor, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion,
          maxLength=MAX_LENGTH):

    # 初始化隐层张量
    encoderHidden = encoder.initHidden()

    # 编码器和解码器优化器梯度归0
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    # 根据源文本和目标文本张量获得对应的长度
    inputLength = inputTensor.size(0)
    targetLength = targetTensor.size(0)

    # 初始化编码器输出张量，形状是max_length x encoder.hidden_size的0张量
    encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize)

    # 初始设置损失为0
    loss = 0

    # 循环遍历输入张量索引
    for ei in range(inputLength):
        # 根据索引从input_tensor取出对应的单词的张量表示，和初始化隐层张量一同传入encoder对象中
        encoderOutput, encoderHidden = encoder(
            inputTensor[ei], encoderHidden)
        # 将每次获得的输出encoder_output(三维张量), 使用[0, 0]降两维变成向量依次存入到encoder_outputs
        # 这样encoder_outputs每一行存的都是对应的句子中每个单词通过编码器的输出结果
        encoderOutputs[ei] = encoderOutput[0, 0]

    # 初始化解码器的第一个输入，即起始符
    decoderInput = torch.tensor([[SOS_TOKEN]])

    # 初始化解码器的隐层张量即编码器的隐层输出
    decoderHidden = encoderHidden

    # teacherForcing
    useTeacherForcing = True if random.random() < TEACHER_FORCING_RATIO else False

    # 如果使用teacherForcing
    if useTeacherForcing:
        # 循环遍历目标张量索引
        for di in range(targetLength):
            # 将decoder_input, decoderHidden, encoder_outputs即attention中的QKV,
            # 传入解码器对象, 获得decoder_output, decoderHidden, decoderAttention
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)
            # 因为使用了teacher_forcing, 无论解码器输出的decoder_output是什么, 我们都只
            # 使用‘正确的答案’，即target_tensor[di]来计算损失
            loss += criterion(decoderOutput, targetTensor[di])
            # 并强制将下一次的解码器输入设置为‘正确的答案’
            decoderInput = targetTensor[di]

    else:
        # 如果不使用teacher_forcing
        # 仍然遍历目标张量索引
        for di in range(targetLength):
            # 将decoder_input, decoderHidden, encoder_outputs传入解码器对象
            # 获得decoder_output, decoderHidden, decoderAttention
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)
            # 只不过这里我们将从decoder_output取出答案
            topv, topi = decoderOutput.topk(1)
            # 损失计算仍然使用decoder_output和target_tensor[di]
            loss += criterion(decoderOutput, targetTensor[di])
            # 最后如果输出值是终止符，则循环停止
            if topi.squeeze().item() == EOS_TOKEN:
                break
            # 否则，并对topi降维并分离赋值给decoder_input以便进行下次运算
            # 这里的detach的分离作用使得这个decoder_input与模型构建的张量图无关，相当于全新的外界输入
            decoderInput = topi.squeeze().detach()

    # 误差进行反向传播
    loss.backward()
    # 编码器和解码器进行优化即参数更新
    encoderOptimizer.step()
    decoderOptimizer.step()

    # 最后返回平均损失
    return loss.item() / targetLength

# 获得每次打印的训练耗时
def timeSince(since):
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

# ----------------------------- 构建模型训练函数, 并进行训练 END -----------------------------


# ----------------------------- 构建模型训练函数, 并进行训练 START -----------------------------


# encoder, decoder: 编码器和解码器对象
# nIters: 总迭代步数
# printEvery: 打印日志间隔
# plotEvery: 绘制损失曲线间隔
# learningRate: 学习率
# 训练迭代函数
def trainIters(srcLang, targetLang, pairs, encoder, decoder, nIters, printEvery=1000, plotEvery=100, learningRate=0.01):

    # 获得训练开始时间戳
    start = time.time()

    # 每个损失间隔的平均损失保存列表，用于绘制损失曲线
    plotLosses = []

    # 每个打印日志间隔的总损失，初始为0
    printLossTotal = 0

    # 每个绘制损失间隔的总损失，初始为0
    plotLossTotal = 0

    # 使用预定义的SGD作为优化器，将参数和学习率传入其中
    encoderOptimizer = optim.SGD(encoder.parameters(), lr=learningRate)
    decoderOptimizer = optim.SGD(decoder.parameters(), lr=learningRate)

    # 损失函数：NLL
    criterion = nn.NLLLoss()

    # 根据设置迭代步进行循环
    for iter in range(1, nIters + 1):
        # 每次从语言对列表中随机取出一条作为训练语句
        trainingPair = tensorsFromPair(srcLang, targetLang, random.choice(pairs))
        # 分别从trainingPair中取出输入张量和目标张量
        inputTensor = trainingPair[0]
        targetTensor = trainingPair[1]

        # 通过train函数获得模型运行的损失
        loss = train(inputTensor, targetTensor, encoder,
                     decoder, encoderOptimizer, decoderOptimizer, criterion)
        # 将损失进行累和
        printLossTotal += loss
        plotLossTotal += loss

        # 当迭代步达到日志打印间隔时
        if iter % printEvery == 0:
            # 通过总损失除以间隔得到平均损失
            printLossAvg = printLossTotal / printEvery
            # 将总损失归0
            printLossTotal = 0
            # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
            print('%s (%d %d%%) %.4f' % (timeSince(start),
                                         iter, iter / nIters * 100, printLossAvg))

        # 当迭代步达到损失绘制间隔时
        if iter % plotEvery == 0:
            # 通过总损失除以间隔得到平均损失
            plotLossAvg = plotLossTotal / plotEvery
            # 将平均损失装进plotLosses列表
            plotLosses.append(plotLossAvg)
            # 总损失归0
            plotLossTotal = 0

    # 绘制损失曲线
    plt.figure()
    plt.plot(plotLosses)
    plt.savefig("./seq2seqLoss.png")

srcLangTrain, targetLangTrain, pairsTrain = prepareData(TRAIN_SRC, TRAIN_TARGET)
hiddenSize = 256
encoder = EncoderRNN(srcLangTrain.n_words, hiddenSize)
attentionDecoder = AttentionDecoderRNN(hiddenSize, targetLangTrain.n_words, dropoutP=0.1)
nIters = 75000
printEvery = 1000
trainIters(srcLangTrain, targetLangTrain, pairsTrain, encoder, attentionDecoder, nIters, printEvery=printEvery)

# ----------------------------- 构建模型训练函数, 并进行训练 END -----------------------------

# ----------------------------- 构建模型评估函数, 并进行测试以及Attention效果分析 START -----------------------------

# encoder, decoder: 编码器和解码器对象
# sentence: 需要评估的句子
# maxLength: 句子的最大长度
# 评估函数
def validatePerSentence(srcLang, targetLang, encoder, decoder, sentence, maxLength=MAX_LENGTH):
    # 不进行梯度计算
    with torch.no_grad():
        # 获取输入张量
        inputTensor = tensorFromSentence(srcLang, sentence, 'en')
        # 获得句子长度
        inputLength = inputTensor.size()[0]
        # 初始化编码器隐层张量
        encoderHidden = encoder.initHidden()
        # 初始化编码器输出张量
        encoderOutputs = torch.zeros(maxLength, encoder.hiddenSize)
        # 循环遍历输入张量索引
        for ei in range(inputLength):
            # 根据索引从inputTensor取出对应的单词的张量表示，和初始化隐层张量一同传入encoder对象中
            encoderOutput, encoderHidden = encoder(inputTensor[ei],
                                                     encoderHidden)
            # 将每次获得的输出encoderOutput(三维张量), 使用[0, 0]降两维变成向量依次存入到encoderOutputs
            # 这样encoderOutputs每一行存的都是对应的句子中每个单词通过编码器的输出结果
            encoderOutputs[ei] += encoderOutput[0, 0]
        # 初始化解码器的第一个输入，即起始符
        decoderInput = torch.tensor([[SOS_TOKEN]])
        # 初始化解码器的隐层张量即编码器的隐层输出
        decoderHidden = encoderHidden
        # 初始化预测的词汇列表
        decodedWords = []
        # 初始化attention张量
        decoderAttentions = torch.zeros(maxLength, maxLength)
        # 开始循环解码
        for di in range(maxLength):
            # 将decoderInput, decoderHidden, encoderOutputs传入解码器对象
            # 获得decoderOutput, decoderHidden, decoderAttention
            decoderOutput, decoderHidden, decoderAttention = decoder(
                decoderInput, decoderHidden, encoderOutputs)
            # 取所有的attention结果存入初始化的attention张量中
            decoderAttentions[di] = decoderAttention.data
            # 从解码器输出中获得概率最高的值及其索引对象
            topv, topi = decoderOutput.data.topk(1)
            # 从索引对象中取出它的值与结束标志值作对比
            if topi.item() == EOS_TOKEN:
                # 如果是结束标志值，则将结束标志装进decodedWords列表，代表翻译结束
                decodedWords.append('<EOS>')
                # 循环退出
                break
            else:
                # 否则，根据索引找到它在输出语言的index2word字典中对应的单词装进decodedWords
                decodedWords.append(targetLang.index2word[topi.item()])
            # 最后将本次预测的索引降维并分离赋值给decoderInput，以便下次进行预测
            decoderInput = topi.squeeze().detach()
        # 返回结果decodedWords， 以及完整注意力张量, 把没有用到的部分切掉
        return decodedWords, decoderAttentions[:di + 1]

# encoder, decoder: 编码器和解码器对象
# n: 测试数
# 随机选择指定数量的数据进行评估
def validate(srcLang, targetLang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        outputWords, attentions = validatePerSentence(srcLang, targetLang, encoder, decoder, pair[0])
        outputSentence = ' '.join(outputWords)
        print('[',i,']','输入：', pair[0])
        print('[',i,']','目标输出：', pair[1])
        print('[',i,']','模型输出：', outputSentence)
        print('')

srcLangValidation, targetLangValidation, pairsValidation = prepareData(VALIDATION_SRC, VALIDATION_TARGET)
print(targetLangValidation.index2word)
srcLangTest, targetLangTest, pairsTest = prepareData(TEST_SRC, TEST_TARGET)
validate(srcLangValidation, targetLangValidation, pairsValidation, encoder, attentionDecoder)

# ----------------------------- 构建模型评估函数, 并进行测试以及Attention效果分析 END -----------------------------
