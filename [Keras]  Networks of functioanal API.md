<!-- TOC -->

- [什么是函数式(functioanal)API](#什么是函数式functioanalapi)
- [构建网络的步骤](#构建网络的步骤)
  - [第一步 定义输入](#第一步-定义输入)
  - [第二步 连接不同的网络层](#第二步-连接不同的网络层)
  - [第三步 创建模型](#第三步-创建模型)
- [函数式网络模板](#函数式网络模板)
  - [多层感知器(Multilayer Perceptron)](#多层感知器multilayer-perceptron)
  - [卷积神经网络(CNN)](#卷积神经网络cnn)
  - [递归神经网络(RNN)](#递归神经网络rnn)
  - [共享输入层(Shared Input Layer)](#共享输入层shared-input-layer)
  - [共享特征提取层(Shared Feature Extraction Layer)](#共享特征提取层shared-feature-extraction-layer)
  - [多输入模型](#多输入模型)
  - [多输出模型](#多输出模型)
  - [多输入多输出模型](#多输入多输出模型)
  - [Unet](#unet)

<!-- /TOC -->
总结一下几种常用网络的函数式（functioanal）API，不会有人嗨搁那用序贯式(sequential)API搭建网络的吧，不会吧不会吧~
主要参考：

[开始使用 Keras 函数式 API](https://keras.io/zh/getting-started/functional-api-guide/)

[Keras函数式编程（举例超详细）](https://blog.csdn.net/ting0922/article/details/94437540)
#  什么是函数式(functioanal)API
正如帅帅所说，Keras可以使用以下3种方式构建模型：

1. 使用Sequential按层顺序构建模型
2. 使用函数式API构建任意结构模型
3. 继承Model基类构建自定义模型。

序贯(sequential)API允许你为大多数问题逐层堆叠创建模型。虽然说对很多的应用来说，这样的一个手法很简单也解决了很多深度学习网络结构的构建，但是它也有限制－它不允许你创建模型有共享层或有多个输入或输出的网络。

Keras中的函数式(functioanal)API是创建网络模型的另一种方式，它提供了更多的灵活性，包括创建更复杂的模型。

Keras 函数式 API 是定义复杂模型（如多输出模型、有向无环图，或具有共享层的模型）的方法。

简单的说，复杂的网络都用函数式构建。

#  构建网络的步骤
Keras函数式(functional)API为构建网络模型提供了更为灵活的方式。
它允许你定义多个输入或输出模型以及共享图层的模型。除此之外，它允许你定义动态(ad-hoc)的非周期性(acyclic)网络图。
模型是通过创建层的实例(layer instances)并将它们直接相互连接成对来定义的，然后定义一个模型(model)来指定那些层是要作为这个模型的输入和输出。
##  第一步 定义输入
与Sequential模型不同，你必须创建独立的Input层物件的instance并定义输入数据张量的维度形状(tensor shape)。
输入层采用一个张量形状参数(tensor shape),它是一个tuple,用于宣告输入张量的维度。
例如：我们要把MNIST的每张图像(28 * 28)打平成一个一维(784)的张量作为一个多层感知机(MLP)的Input。
```python
from keras.layers import Input

mnist_input = Input(shape=(784,))

```
##  第二步 连接不同的网络层
模型中的神经层是成对连接的，就像是一个乐高积木一样有一面是凸的一面是凹的，一个神经层的输出会接到另一个神经层的输入。
这是通过在定义每个新神经层时指定输入的来源来完成的。使用括号表示法，以便在创建图层之后，指定作为输入的神经层。
我们用一个简短的例子来说明这一点。我们可以像上面那样创建输入层，然后创建一个隐藏层作为密集层，它接收来自输入层的输入。

```python
from keras.layers import Input
from keras.layers import Dense

mnist_input = Input(shape=(784,))
hidden = Dense(512)(mnist_input)

```
看见`hidden = Dense(512)(mnist_input)`中这个括号`(mnist_input)`没有，他就是你要链接的上一层，而`Dense(512)`则是这一层的结构，整个网络就是这么这一层接着上一层写下来的。

正是这种逐层连接的方式赋予功能性(functional)API灵活性。你可以看到开始一些动态的神经网络是多么容易。

##  第三步 创建模型
在创建所有模型图层并将它们连接在一起后，你必须定义一个模型(Model)物件的instance。
与Sequential API一样，这个模型是你可以用于总结(summarize),拟合(fit)，评估(evaluate)和预测(predict)。
Keras提供了一个Model类别，你可以使用它从创建的图层创建模型的instance。它会要求你只指定整个模型的第一个输入层和最后一个输出层。例如:

```python
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

mnist_input = Input(shape=(784,))
hidden = Dense(512)(mnist_input)

model = Model(inputs=mnist_input,outputs=hidden)

```
#  函数式网络模板
叭叭那么多有锤子用，来几个例子模板才是最有用的，研究完几个例子还能不会？
##  多层感知器(Multilayer Perceptron)
从最简单的开始
让我们来定义一个多类别分类(multi-class classification)的多层感知器(MLP)模型。
该模型有784个输入，３个隐藏层，512,216和128个隐藏神经元，输出层有10个输出。
在每个隐藏层中使用relu激活函数，并且在输出层中使用softmax激活函数进行多类别分类。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606223829224.png)
```python
# 多层感知器MLP模型

from keras.models import Model
from keras.layers import Input,Dense
from keras.utils import plot_model

import matplotlib.pyplot as plt
from IPython.display import Image

mnist_input = Input(shape=(784,),name='input')
hidden1 = Dense(512,activation='relu',name='hidden1')(mnist_input)
hidden2 = Dense(216,activation='relu',name='hidden2')(hidden1)
hidden3 = Dense(128,activation='relu',name='hidden3')(hidden2)
output = Dense(10,activation='softmax',name='output')(hidden3)

model = Model(inputs=mnist_input,outputs=output)

# 打印网络结构
model.summary()

# 产生网络拓补图
plot_model(model,to_file='multilayer_perceptron_graph.png')

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 784)               0
_________________________________________________________________
hidden1 (Dense)              (None, 512)               401920
_________________________________________________________________
hidden2 (Dense)              (None, 216)               110808
_________________________________________________________________
hidden3 (Dense)              (None, 128)               27776
_________________________________________________________________
output (Dense)               (None, 10)                1290
=================================================================
Total params: 541,794
Trainable params: 541,794
Non-trainable params: 0

```
##  卷积神经网络(CNN)
我们将定义一个用于图像分类的卷积神经网络(Convolutional neural network)。
该模型接收灰阶的28 * 28图像作为输入，然后有一个作为特征提取器的两个卷积和池化层的序列，然后是一个完全连接层来解释特征，并且具有用于10类预测的softmax激活的输出层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606224012154.png)

```python
# 卷积神经网络(CNN)
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.utils import plot_model

mnist_input = Input(shape=(28,28,1), name='input')

conv1 = Conv2D(128,kernel_size=4,activation='relu',name='conv1')(mnist_input)
pool1 = MaxPool2D(pool_size=(2,2),name='pool1')(conv1)

conv2 = Conv2D(64,kernel_size=4,activation='relu',name='conv2')(pool1)
pool2 = MaxPool2D(pool_size=(2,2),name='pool2')(conv2)

hidden1 = Dense(64,activation='relu',name='hidden1')(pool2)
output = Dense(10,activation='softmax',name='output')(hidden1)
model = Model(inputs=mnist_input,outputs=output)

# 打印网络结构
model.summary()

# 产生网络拓补图
plot_model(model,to_file='convolutional_neural_network.png')

# 秀出网络拓补图
Image('convolutional_neural_network.png')
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 28, 28, 1)         0
_________________________________________________________________
conv1 (Conv2D)               (None, 25, 25, 128)       2176
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 12, 12, 128)       0
_________________________________________________________________
conv2 (Conv2D)               (None, 9, 9, 64)          131136
_________________________________________________________________
pool2 (MaxPooling2D)         (None, 4, 4, 64)          0
_________________________________________________________________
hidden1 (Dense)              (None, 4, 4, 64)          4160
_________________________________________________________________
output (Dense)               (None, 4, 4, 10)          650
=================================================================
Total params: 138,122
Trainable params: 138,122
Non-trainable params: 0

```
##  递归神经网络(RNN)
我们将定义一个长短期记忆(LSTM)递归神经网络用于图像分类。
该模型预期一个特征的784个时间步骤作为输入。该模型具有单个LSTM隐藏层以从序列中提取特征，接着是完全连接的层来解释LSTM输出，接着是用于进行10类别预测的输出层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606224237941.png)

```python
# 递归神经网络(RNN)
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.recurrent import LSTM
from keras.utils import plot_model

mnist_input = Input(shape=(784,1),name='input') # 把每一个像素想成是一序列有前后关系的time_steps
lstm1 = LSTM(128,name='lstm')(mnist_input)
hidden1 = Dense(128,activation='relu',name='hidden1')(lstm1)
output = Dense(10,activation='softmax',name='output')(hidden1)
model = Model(inputs=mnist_input,outputs=output)

# 打印网络结构
model.summary()

# 产生网络拓补图
plot_model(model,to_file='recurrent_neural_network.png')

# 秀出网络拓补图
Image('recurrent_neural_network.png')
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 784, 1)            0
_________________________________________________________________
lstm (LSTM)                  (None, 128)               66560
_________________________________________________________________
hidden1 (Dense)              (None, 128)               16512
_________________________________________________________________
output (Dense)               (None, 10)                1290
=================================================================
Total params: 84,362
Trainable params: 84,362
Non-trainable params: 0

```
##  共享输入层(Shared Input Layer)
我们定义具有不同大小的内核的多个卷积层来解释图像输入。
该模型使用28　×　28　像素的灰阶图像。有两个CNN特征提取子模型共享这个输入；第一个具有４的内核大小和第二个８的内核大小。这些特征提取子模型的输出被平坦化(flatten)为向量(vector),并且被串成一个长向量；然后被传递到完全连接的层以用于最终输出层之前进行10类别预测。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606224512800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NzM2NTA0,size_16,color_FFFFFF,t_70)

```python
# 共享输入层
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import concatenate
from keras.utils import plot_model

# 输入层
mnist_input = Input(shape=(28,28,1),name='input')

# 第一个特征提取层
conv1 = Conv2D(32,kernel_size=4,activation='relu',name='conv1')(mnist_input) # <- 看这里
pool1 = MaxPool2D(pool_size=(2,2),name='pool1')(conv1)
flat1 = Flatten()(pool1)

# 第二个特征提取层
conv2 = Conv2D(16,kernel_size=8,activation='relu',name='conv2')(mnist_input) # <- 看这里
pool2 = MaxPool2D(pool_size=(2,2),name='pool2')(conv2)
flat2 = Flatten()(pool2)

# 把这两个特征提取层的结果拼接起来
merge = concatenate([flat1,flat2])

# 进行全连接层
hidden1 = Dense(64,activation='relu',name='hidden1')(merge)

# 输出层
output = Dense(10,activation='softmax',name='output')(hidden1)

# 以model来组合整个网络
model = Model(inputs=mnist_input,outputs=output)

# 打印网络结构
model.summary()

# 网络结构可视化
plot_model(model,to_file='shared_input_layer.png')

# 秀出网络拓补图
Image('shared_input_layer.png')
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              (None, 28, 28, 1)    0
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 25, 25, 32)   544         input[0][0]
__________________________________________________________________________________________________
conv2 (Conv2D)                  (None, 21, 21, 16)   1040        input[0][0]
__________________________________________________________________________________________________
pool1 (MaxPooling2D)            (None, 12, 12, 32)   0           conv1[0][0]
__________________________________________________________________________________________________
pool2 (MaxPooling2D)            (None, 10, 10, 16)   0           conv2[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 4608)         0           pool1[0][0]
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 1600)         0           pool2[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 6208)         0           flatten_1[0][0]
                                                                 flatten_2[0][0]
__________________________________________________________________________________________________
hidden1 (Dense)                 (None, 64)           397376      concatenate_1[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 10)           650         hidden1[0][0]
==================================================================================================
Total params: 399,610
Trainable params: 399,610
Non-trainable params: 0

```
##  共享特征提取层(Shared Feature Extraction Layer)
我们将使用两个并行子模型来解释用于序列分类的LSTM特征提取器的输出。
该模型的输入是一个特征的784个时间步长。具有１０个存储单元的LSTM层解释这个序列。第一种解释模型是浅层单连通层，第二层是深层３层模型。两个解释模型的输出连接成一个长向量，传递给用于进行１０类别分类预测的输出层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606224710103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NzM2NTA0,size_16,color_FFFFFF,t_70)

```python
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.utils import plot_model

# 输入层
mnist_input = Input(shape=(784,1),name='input') # 把每一个像素想成是一序列有前后关系的time_steps

# 特征提取层
extract1 = LSTM(128,name='lstm1')(mnist_input)

# 第一个解释层(浅层单连通层)
interp1 = Dense(10,activation='relu',name='interp1')(extract1) # <- 看这里

# 第二个解释层(深层3层模型)
interp21 = Dense(64,activation='relu',name='interp21')(extract1) # <- 看这里
interp22 = Dense(32,activation='relu',name='interp22')(interp21)
interp23 = Dense(10,activation='relu',name='interp23')(interp22)

# 把两个特征提取层的结果拼起来
merge = concatenate([interp1,interp23],name='merge')

# 输出层
output = Dense(10,activation='softmax',name='output')(merge)

# 以Ｍodel来组合整个网络
model = Model(inputs=mnist_input,outputs=output)

# 打印网络结构
model.summary()

# 可视化
plot_model(model,to_file='shared_feature_extractor.png')

# 秀出网络拓补图
Image('shared_feature_extractor.png')
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              (None, 784, 1)       0
__________________________________________________________________________________________________
lstm1 (LSTM)                    (None, 128)          66560       input[0][0]
__________________________________________________________________________________________________
interp21 (Dense)                (None, 64)           8256        lstm1[0][0]
__________________________________________________________________________________________________
interp22 (Dense)                (None, 32)           2080        interp21[0][0]
__________________________________________________________________________________________________
interp1 (Dense)                 (None, 10)           1290        lstm1[0][0]
__________________________________________________________________________________________________
interp23 (Dense)                (None, 10)           330         interp22[0][0]
__________________________________________________________________________________________________
merge (Concatenate)             (None, 20)           0           interp1[0][0]
                                                                 interp23[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 10)           210         merge[0][0]
==================================================================================================
Total params: 78,726
Trainable params: 78,726
Non-trainable params: 0

```
##  多输入模型
我们将开发一个图像分类模型，将图像的两个版本作为输入，每个图像的大小不同。特别是一个灰阶的64 * 64版本和一个32 * 32 彩色版本。分离的特征提取CNN模型对每个模型进行操作，然后将两个模型的结果连接起来进行解释和最终的预测。
请注意在创建Model()实例(instance)时，我们将两个输入层定义为一个数组(array)。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606225018202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NzM2NTA0,size_16,color_FFFFFF,t_70)

```python
# 多输入模型
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers.merge import concatenate
from keras.utils import plot_model
from IPython.display import Image

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'  # 安装graphviz的路径

# 第一个输入层
img_gray_bigsize = Input(shape=(64,64,1),name='img_gray_bigsize')
conv11 = Conv2D(32,kernel_size=4,activation='relu',name='conv11')(img_gray_bigsize)
pool11 = MaxPool2D(pool_size=(2,2),name='pool11')(conv11)
conv12 = Conv2D(16,kernel_size=4,activation='relu',name='conv12')(pool11)
pool12 = MaxPool2D(pool_size=(2,2),name='pool12')(conv12)
flat1 = Flatten()(pool12)

# 第二个输入层
img_rgb_smallsize = Input(shape=(32,32,3),name='img_rgb_bigsize')
conv21 = Conv2D(32,kernel_size=4,activation='relu',name='conv21')(img_rgb_smallsize)
pool21 = MaxPool2D(pool_size=(2,2),name='pool21')(conv21)
conv22 = Conv2D(16,kernel_size=4,activation='relu',name='conv22')(pool21)
pool22 = MaxPool2D(pool_size=(2,2),name='pool22')(conv22)
flat2 = Flatten()(pool22)

# 把两个特征提取层的结果拼起来
merge = concatenate([flat1,flat2])

# 用隐藏的全连接层来解释特征
hidden1 = Dense(128,activation='relu',name='hidden1')(merge)
hidden2 = Dense(64,activation='relu',name='hidden2')(hidden1)

# 输出层
output = Dense(10,activation='softmax',name='output')(hidden2)
# 以Model来组合整个网络
model = Model(inputs=[img_gray_bigsize,img_rgb_smallsize],outputs=output)

# 打印网络结构
model.summary()
# 可视化
plot_model(model,to_file='multiple_inputs.png')

# 秀出网络拓补图
Image('multiple_inputs.png')
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
img_gray_bigsize (InputLayer)   (None, 64, 64, 1)    0
__________________________________________________________________________________________________
img_rgb_bigsize (InputLayer)    (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv11 (Conv2D)                 (None, 61, 61, 32)   544         img_gray_bigsize[0][0]
__________________________________________________________________________________________________
conv21 (Conv2D)                 (None, 29, 29, 32)   1568        img_rgb_bigsize[0][0]
__________________________________________________________________________________________________
pool11 (MaxPooling2D)           (None, 30, 30, 32)   0           conv11[0][0]
__________________________________________________________________________________________________
pool21 (MaxPooling2D)           (None, 14, 14, 32)   0           conv21[0][0]
__________________________________________________________________________________________________
conv12 (Conv2D)                 (None, 27, 27, 16)   8208        pool11[0][0]
__________________________________________________________________________________________________
conv22 (Conv2D)                 (None, 11, 11, 16)   8208        pool21[0][0]
__________________________________________________________________________________________________
pool12 (MaxPooling2D)           (None, 13, 13, 16)   0           conv12[0][0]
__________________________________________________________________________________________________
pool22 (MaxPooling2D)           (None, 5, 5, 16)     0           conv22[0][0]
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 2704)         0           pool12[0][0]
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 400)          0           pool22[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 3104)         0           flatten_3[0][0]
                                                                 flatten_4[0][0]
__________________________________________________________________________________________________
hidden1 (Dense)                 (None, 128)          397440      concatenate_2[0][0]
__________________________________________________________________________________________________
hidden2 (Dense)                 (None, 64)           8256        hidden1[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 10)           650         hidden2[0][0]
==================================================================================================
Total params: 424,874
Trainable params: 424,874
Non-trainable params: 0

```
##  多输出模型
我们将开发一个模型，进行两种不同模型的预测。给定一个特征的784个时间步长的输入序列，该模型将对该序列进行分类并输出具有相同长度的新序列。
LSTM层解释输入序列并返回每个时间步长的隐藏状态。第一个输出模型创建一个堆叠的LSTM，解释这些特征，并进行多类别预测。第二个输出模型使用相同的输出层对每个输入时间步进行多类别预测。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606225153183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NzM2NTA0,size_16,color_FFFFFF,t_70)

```python
# 多输出模型
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model

# 输入层
mnist_input = Input(shape=(784,1),name='input') # 吧每一个像素想成是一序列有前后关系的time_steps

# 特征提取层
extract = LSTM(64,return_sequences=True,name='extract')(mnist_input)

# 分类输出
class11 = LSTM(32,name='class11')(extract)
class12 = Dense(32,activation='relu',name='class12')(class11)
output1 = Dense(10,activation='softmax',name='output1')(class12)

# 序列输出
output2 = TimeDistributed(Dense(10,activation='softmax'),name='output2')(extract)

# 以Model来组合整个网络
model = Model(inputs=mnist_input,outputs=[output1,output2])

# 打印网络结构
model.summary()

# plot_model可视化
plot_model(model,to_file='multiple_outputs.png')
# 秀出拓补图
Image('multiple_outputs.png')
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              (None, 784, 1)       0
__________________________________________________________________________________________________
extract (LSTM)                  (None, 784, 64)      16896       input[0][0]
__________________________________________________________________________________________________
class11 (LSTM)                  (None, 32)           12416       extract[0][0]
__________________________________________________________________________________________________
class12 (Dense)                 (None, 32)           1056        class11[0][0]
__________________________________________________________________________________________________
output1 (Dense)                 (None, 10)           330         class12[0][0]
__________________________________________________________________________________________________
output2 (TimeDistributed)       (None, 784, 10)      650         extract[0][0]
==================================================================================================
Total params: 31,348
Trainable params: 31,348
Non-trainable params: 0

```
##  多输入多输出模型
以下是函数式 API 的一个很好的例子：具有多个输入和输出的模型。函数式 API 使处理大量交织的数据流变得容易。

来考虑下面的模型。我们试图预测 Twitter 上的一条新闻标题有多少转发和点赞数。模型的主要输入将是新闻标题本身，即一系列词语，但是为了增添趣味，我们的模型还添加了其他的辅助输入来接收额外的数据，例如新闻标题的发布的时间等。 该模型也将通过两个损失函数进行监督学习。较早地在模型中使用主损失函数，是深度学习模型的一个良好正则方法。

模型结构如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606225413579.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NzM2NTA0,size_16,color_FFFFFF,t_70)
让我们用函数式 API 来实现它。

主要输入接收新闻标题本身，即一个整数序列（每个整数编码一个词）。 这些整数在 1 到 10,000 之间（10,000 个词的词汇表），且序列长度为 100 个词。

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 "name" 参数来命名任何层。
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Embedding 层将输入序列编码为一个稠密向量的序列，
# 每个向量维度为 512。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM 层把向量序列转换成单个向量，
# 它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)
```
在这里，我们插入辅助损失，使得即使在模型主损失很高的情况下，LSTM 层和 Embedding 层都能被平稳地训练。

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```
此时，我们将辅助输入数据与 LSTM 层的输出连接起来，输入到模型中：

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```
然后定义一个具有两个输入和两个输出的模型：

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```
现在编译模型，并给辅助损失分配一个 0.2 的权重。如果要为不同的输出指定不同的 loss_weights 或 loss，可以使用列表或字典。 在这里，我们给 loss 参数传递单个损失函数，这个损失将用于所有的输出。

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```
我们可以通过传递输入数组和目标数组的列表来训练模型：

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```
由于输入和输出均被命名了（在定义时传递了一个 name 参数），我们也可以通过以下方式编译模型：

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# 然后使用以下方式训练：
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```
##  Unet

[U-net源码讲解（Keras）](https://blog.csdn.net/mieleizhi0522/article/details/82217677)

详解：
1.输入是572x572的，但是输出变成了388x388，这说明经过网络以后，输出的结果和原图不是完全对应的，这在计算loss和输出结果都可以得到体现。
2.蓝色箭头代表3x3的卷积操作，并且stride是1，padding策略是vaild，因此，每个该操作以后，featuremap的大小会减2。
3.红色箭头代表2x2的maxpooling操作，需要注意的是，此时的padding策略也是vaild（same 策略会在边缘填充0，保证featuremap的每个值都会被取到，vaild会忽略掉不能进行下去的pooling操作，而不是进行填充），这就会导致如果pooling之前featuremap的大小是奇数，那么就会损失一些信息 。
4.绿色箭头代表2x2的反卷积操作，这个只要理解了反卷积操作，就没什么问题，操作会将featuremap的大小乘2。
5.灰色箭头表示复制和剪切操作，可以发现，在同一层左边的最后一层要比右边的第一层要大一些，这就导致了，想要利用浅层的feature，就要进行一些剪切，也导致了最终的输出是输入的中心某个区域。
6.输出的最后一层，使用了1x1的卷积层做了分类。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200606230254866.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NzM2NTA0,size_16,color_FFFFFF,t_70)
收缩路径就是常规的卷积网络，它包含重复的2个3x3卷积，紧接着是一个RELU，一个max pooling（步长为2），用来降采样，每次降采样我们都将feature channel减半。
```python
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
```

```python
def unet(pretrained_weights = None,input_size = (572,572,1)):
################################################提取特征############################################################
    #左一
    inputs = Input(input_size)#572*572
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#570*570*64
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#568*568*64

    #左二
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#284*284*64
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#282*282*128
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)#280*280*128

    #左三
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#140*140*128
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#138*138*256
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)#136*136*256

    #左四
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#68*68*256
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)#66*66*512
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)#64*64*512
    drop4 = Dropout(0.5)(conv4)

    #左五
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)#32*32*512
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)#30*30*1024
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)#28*28*1024
    drop5 = Dropout(0.5)(conv5)

################################################上采样部分##################################################################
    #右四
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))#56*56*512
    #上采样之后再进行卷积，相当于转置卷积操作！
    merge6 = concatenate([drop4,up6],axis = 3)#左四连右四56*56*1024
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)#54*54*512
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)#52*52*512

    #右三
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))#104*104*256
    merge7 = concatenate([conv3,up7],axis = 3)#左三连右三104*104*512
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)#102*102*256
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)#100*100*256

    #右二
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))#200*200*128
    merge8 = concatenate([conv2,up8],axis = 3)#左二连右二200*200*256
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)#198*198*128
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)#196*196*128

    #右一
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))#392*392*64
    merge9 = concatenate([conv1,up9],axis = 3)#左一连右一392*392*128
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)#390*390*64
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)#388*388*64
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)#388*388*2
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)#我怀疑这个sigmoid激活函数是多余的，因为在后面的loss中用到的就是二进制交叉熵，包含了sigmoid

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])#模型执行之前必须要编译https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
    #利用二进制交叉熵，也就是sigmoid交叉熵，metrics一般选用准确率，它会使准确率往高处发展
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
```
