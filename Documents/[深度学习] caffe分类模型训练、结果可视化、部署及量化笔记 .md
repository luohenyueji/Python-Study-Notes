本文为本人caffe分类网络训练、结果可视化、部署及量化具体过程的心得笔记。caffe目前官方已经停止支持了，但是caffe是目前工业落地最常用的深度学习框架，用的人挺多。其实主要怕自己忘了，弄个备份，弄caffe很久了，怕不用东西都忘了，但是本文主要是讲述caffe下的分类网络。caffe默认已经配置好了，而且尽可能是linux系统，本文基于ubuntu18系统。如果有错误，希望积极指正。
# 1 训练
## 1.1 数据准备
首先在caffe/data路径建立example_data文件夹，在example_data里建立三个文件夹。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808153723769.png)
train文件为训练文件数据，val为验证文件数据，dataSet为最后生成caffe所用数据存放文件夹。这里准备五类数据，分别放在文件夹0-4。标号必须为0到4。val文件夹也是一样的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808154146203.png)
0到4文件夹存放各类图像，各类图像编号类似如下：![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808164240294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
然后调用label.py创建数据集标签名，分别在train目录和val目录生成数据集标签文件，创建数据集的txt文件，如果是训练集命名为train.txt，如果是验证集命名为val.txt。label.py代码如下：
```python
import os 
# 各类分类文件夹名
dealPaths=['0','1','2','3','4']
# 创建数据集的txt文件，如果是训练集命名为train.txt如果是验证集命名为val.txt
imageData=open('imageData.txt','w')

for dealPath in dealPaths:
    for filename in os.listdir(dealPath):
        imageData.write(filename+' '+dealPath+"\n")
        
imageData.close()
```
生成的数据标签文件如下，这个数据标签文件包含所有训练集图像数据标签，图像必须是jpg文件。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808164923269.png)
然后将所有单个分类也就是0到4文件夹里面的数据放入一个文件夹。这里通过imageMove.py将所有图像移动到自己设定的set文件夹。但是如果是训练数据集，将set文件夹重命名为imageTrain，然后上一步得到的train.txt文件移动到imageTrain里面，然后将验证集数据集得到的set文件夹重命名为imageVal，把上一部得到的val.txt移动到imageVal文件夹里面。
```python

import os
import shutil

movePaths=['0','1','2','3','4']
# 保存的文件夹
dstPath='set'
os.makedirs(dstPath,exist_ok=True)

imageCount=[]

for movePath in movePaths:
    imageCount.append(len(os.listdir(movePath)))
    print('current path is {}'.format(movePath))
    for filename in os.listdir(movePath):
        srcFile=os.path.join(movePath,filename)
        dstFile=os.path.join(dstPath,filename)
        shutil.copy(srcFile,dstFile)
        
if len(os.listdir(dstPath)) == sum(imageCount):
    print('move sucess!')
else:
    print('error')
```
对于建立图像训练集可能会出现jpg文件损坏以及其他问题，一些小工具可以见：
>https://blog.csdn.net/LuohenYJ/article/details/86574451
## 1.2 创建lmdb文件
将上一节得到的imageTrain文件夹和imageVal文件夹移动到dataSet目录。如下所示：
其中create_data.sh用于创建lmbdb文件，make_data_mean.sh用于创建均值文件。首先在caffe根目录下运行create_data.sh文件，命令如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808170404250.png)
create_data.sh脚本具体内容如下，具体要改的参数都有标明，都是在caffe目录下使用相对路径名，绝对路径名容易出错。
```python
#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

# 数据文件根目录
EXAMPLE=data/example_data
# 数据文件目录
DATA=data/example_data/dataSet
# 生成lmdb文件所用到目录
TOOLS=build/tools

# 生成的train.val的lmdb文件保存目录
TRAIN_DATA_ROOT=data/example_data/dataSet/imageTrain/
VAL_DATA_ROOT=data/example_data/dataSet/imageVal/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
# 改一改深度学习模型要求输入图像的大小
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    # 打乱图像
    --shuffle=true \
    $TRAIN_DATA_ROOT \
    # 训练集文件标签
    $DATA/imageTrain/train.txt \
    $EXAMPLE/example_data_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    # 测试集文件标签
    $DATA/imageVal/val.txt \
    $EXAMPLE/example_data_val_lmdb

echo "Done."

```
生成lmdb文件后，然后生成均值文件也就是example_data_mean.binaryproto文件
```python
#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12
# 和前面一样
EXAMPLE=data/example_data
DATA=data/example_data
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/example_data_train_lmdb \
  $DATA/example_data_mean.binaryproto

echo "Done."
```
最后我们会得到均值数据文件，分别是BGR通道训练样本均值。记得把这三个值记一下以后要用到。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808171456251.png)
所得到的文件lmdb和均值文件可以在example_data根目录下找到。训练模型用这三个文件就行了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808171346570.png)
## 1.3 模型训练
训练时还需要solver文件来对参数进行更新， 同时还需要网络结构文件。在caffe/examples下建立example_data文件夹下进行训练。将前面的lmdb文件和均值文件复制到改文件夹下，如下图所示。橙色框选的文件是上步所获得的，红色框选的文件是网络结构参数文件和调参文件。backup是生成用于保存训练模型的文件夹，caffe.log是训练时生成的日志文件。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808192523910.png)
solver文件就用示例的配置alexnet文件。solver文件主要参数搜索就用，其中net值得是模型结构文件路径，训练习惯以caffe为根目录，snapshot_prefix为模型保存目录。
```python
# 网络结构地址
net: "examples/example_data/alexnet_train_val.prototxt"
test_iter: 200
test_interval: 200
base_lr: 0.0001
lr_policy: "step"
gamma: 0.1
stepsize: 1000
display: 100
max_iter: 3000
momentum: 0.9
weight_decay: 0.0005
snapshot: 1000
# 模型保存文件
snapshot_prefix: "examples/example_data/backup/"
solver_mode: GPU
```
至于train_val模型文件，有两部分需要改，第一部分是开头的data文件，需要配置lmdb文件，train和val都配置。其次是均值部分，一种方法是类似配置lmdb文件一样，设定mean_file路径，另外一种是直接给出均值，把前面BGR均值用mean_value:给出。batch_size的修改具体见搜索。lmdb图像输入大小需要和crop_size对应。如下所示：
```python
name: "AlexNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: "examples/example_data/example_data_mean.binaryproto"
    #mean_value: 85.0205
    #mean_value: 85.0205
    #mean_value: 85.0205
  }
  data_param {
    source: "examples/example_data/example_data_train_lmdb"
    batch_size: 4
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "examples/example_data/example_data_mean.binaryproto"
    #mean_value: 85.0205
    #mean_value: 85.0205
    #mean_value: 85.0205
  }
  data_param {
    source: "examples/example_data/example_data_val_lmdb"
    batch_size: 8
    backend: LMDB
  }
}
```
另外一个要改的就是结尾的num_output，将最后一个num_output改为分类个数，本文有5类所以num_output:5
```python
layer {
  name: "conv10"
  type: "Convolution"
  bottom: "fire9/concat"
  top: "conv10"
  convolution_param {
  	# 输出，按照分类个数确定
    num_output: 5
    kernel_size: 1
    weight_filler {
      type: "gaussian"
      mean: 0.0
      std: 0.01
    }
  }
}
layer {
  name: "relu_conv10"
  type: "ReLU"
  bottom: "conv10"
  top: "conv10"
}
layer {
  name: "pool10"
  type: "Pooling"
  bottom: "conv10"
  top: "pool10"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "pool10"
  bottom: "label"
  top: "loss"
  #include {
  #  phase: TRAIN
  #}
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "pool10"
  bottom: "label"
  top: "accuracy"
  #include {
  #  phase: TEST
  #}
}
```
最后进行训练时还是在caffe根目录下输入以下指令。第一个训练指令就是直接训练文件，输出结果打印在命令行。推荐用第二个训练命令，在结果打印在命令行同时，也将结果保存为log日志文件方便以后分析使用。其中-solver表示训练参数，后面跟训练参数文件。如果用到gpu设置 -gpu=0选择用哪个gpu。
```python
./build/tools/caffe train -solver examples/example_data/alexnet_solver.prototxt 

./build/tools/caffe train -solver examples/example_data/alexnet_solver.prototxt 2>&1| tee examples/example_data/caffe.log
```
还有一种训练方式，就是用已有模型进行微调 finetune。如果你不是自己设计模型，这种方式比直接训练要好得多。比如获得squeezenet的solver.prototxt，train_val.prototxt以及模型文件。类似前面直接训练修改solver.protoxt，train_val.prototxt。对于train_val.prototxt修改，只需要修改source处data文件，由于是用别人的模型微调不要改mean值，直接用人家的mean值不要用自己的mean值。
```python
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    crop_size: 227
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "examples/example_data/example_data_train_lmdb"
    batch_size: 32
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    crop_size: 227
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "examples/example_data/example_data_train_lmdb"
    batch_size: 25 #not *iter_size
    backend: LMDB
  }
}
```
对于train_val最后的输出，如果想微调的层就把名字改了，一般都是从后往前改。名字不改的层就不会训练。loss和accuray层不需要改，由于不需要top5输出。accuracy_top5层就被删除了。
```python
layer {
  #原来是conv10
  name: "conv10_example"
  type: "Convolution"
  bottom: "fire9/concat"
  #原来是conv10
  top: "conv10_example"
  convolution_param {
    num_output: 5
    kernel_size: 1
    weight_filler {
      type: "gaussian"
      mean: 0.0
      std: 0.01
    }
  }
}
layer {
  #原来是relu_conv10
  name: "relu_conv10_example"
  type: "ReLU"
  bottom: "conv10_example"
  top: "conv10_example"
}
layer {
  #原来是pool10
  name: "pool10_example"
  type: "Pooling"
  bottom: "conv10_example"
  top: "pool10_example"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "pool10_example"
  bottom: "label"
  top: "loss"
  #include {
  #  phase: TRAIN
  #}
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "pool10_example"
  bottom: "label"
  top: "accuracy"
  #include {
  #  phase: TEST
  #}
}

```
调用参数如下，只是加了weight命令，指向微调网络的模型。
```python
./build/tools/caffe train -solver examples/example_data/solver.prototxt -weights examples/example_data/squeezenet_v1.1.caffemodel 2>&1| tee examples/example_data/caffe.log
```
从训练日志可以看到，训练时会忽视名字改动的层，进行微调。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808202030178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
训练结束后，可以对训练好的模型进行Testing。还是在caffe根目录下输入以下命令进行测试。model表示模型结构参数文件，weights表示模型权重文件路径。会输出当前模型在test数据下准确率和loss。
```python
./build/tools/caffe test -model examples/example_data/train_val.prototxt -weights examples/example_data/backup/solver_iter_100.caffemodel
```
## 1.4 训练部分参考文件
除了官方文件外，参考的内容有：
+ [训练自己的数据集](https://www.cnblogs.com/wktwj/p/6715110.html)
+ [caffe入门](https://blog.csdn.net/cham_3/article/details/72141753)
+ [caffe官方教程](https://blog.csdn.net/hongbin_xu/article/details/79363134)
+ [solver参数设置](https://blog.csdn.net/u014381600/article/details/54428599)
+ [solver参数说明](https://zhuanlan.zhihu.com/p/48462756)
+ [模型微调1](https://blog.csdn.net/u010402786/article/details/70141261)
+ [模型微调2](https://www.cnblogs.com/louyihang-loves-baiyan/p/5038758.html)
+ [模型微调3](https://blog.csdn.net/Angela_qin/article/details/79428987)
+ [模型微调4](https://blog.csdn.net/nongfu_spring/article/details/51514040)
+ [经典网络总结](https://blog.csdn.net/d5224/article/details/77100268)
+ [经典网络模型](https://github.com/SnailTyan/caffe-model-zoo)
+ [caffe猫狗大战训练](https://github.com/mrgloom/kaggle-dogs-vs-cats-solution)

# 2 结果可视化
## 2.1 训练数据展示
 在caffe的训练过程中，需要图形化训练数据结果。caffe中自带了工具显示结果。在examples/example_data下建立analyze文件，然后把训练过程生成的caffe.log移动到该文件夹。然后将caffe根目录tools/extra文件夹下的extract_seconds.py、parse_log.py、parse_log.sh和plot_training_log.py.example文件移动到该文件夹下。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809094044741.png)
 在analyze当前目录下输入以下指令就可以可视化训练日志
 ```python
 ./plot_training_log.py.example 0  save.png caffe.log
 ```
 上面0表示可视化的类型，save.png表示可视化保存的图像名，caffe.log表示日志文件。
 可视化类型具体参数如下。0、1、2等序号表示可视化类型，vs为横坐标参数，vs右边为纵坐标参数
```python
    0: Test accuracy  vs. Iters
    1: Test accuracy  vs. Seconds
    2: Test loss  vs. Iters
    3: Test loss  vs. Seconds
    4: Train learning rate  vs. Iters
    5: Train learning rate  vs. Seconds
    6: Train loss  vs. Iters
    7: Train loss  vs. Seconds
```
绘图结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809095136228.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
这种方法所画出来的图每次都是随机样式且无法定制参数。如果要定制可视化图像，在analyze当前目录下，输入以下参数提取log文件。
```python
./parse_log.py caffe.log ./
```
这样将得到以下两个文件，分别表示训练数据和测试数据
```python
caffe.log.train
caffe.log.test
```
这种方法和前面可视化命令都会得到这两个文件，但是实质内容有所区别。通过parse_log.py得到的文件内容更规则。如caffe.log.test具体内容如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809100419856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
然后在当前文件夹下建立drawCaffe.py文件可视化训练log。drawCaffe.py内容如下：
```python
import pandas as pd
import matplotlib.pyplot as plt

#训练文件
train_log = pd.read_csv("caffe.log.train")
#测试文件
test_log = pd.read_csv("caffe.log.test")


_, ax1 = plt.subplots()
# 可可视化参数NumIters,Seconds,LearningRate,accuracy,loss
ax1.set_title("train loss and test loss")
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.5)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax1.plot(train_log["NumIters"], train_log["accuracy"], alpha=0.5)
ax1.plot(test_log["NumIters"], test_log["accuracy"], 'g')

ax1.set_xlabel('NumIters')
plt.legend(loc='best')

# 保存图像
plt.savefig("save.png", dpi=300)
plt.show()

dfTrain= pd.DataFrame(data=train_log, columns=['NumIters', 'loss', 'accuracy'])
dfTest = pd.DataFrame(data=test_log, columns=['NumIters', 'loss', 'accuracy'])

dfTrain.to_csv('train.csv')
dfTest.to_csv('val.csv')

print("done")

```
通过以上代码能够分析log文件，同时把训练参数结果保存为train.csv和val.csv。其他就是matplotlib美化图像。绘图结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809101320162.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
## 2.2 网络模型可视化
通过netscope可以可视化caffe模型。打开下面链接的网页，打开Editor，将网络结构的prototxt文件复制到网页左侧编辑框后，shift+enter，就可以直接显示网络结构。非常的简单和方便。同时将鼠标选中某层 将可视化其参数。
>http://ethereon.github.io/netscope/quickstart.html
如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809101922599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)

当然这种方式可视化很简单。但是目前深度学习网络框架很多，但不同框架之间可视化网络层方法差别。一个深度学习模型结构可视化神器Netron，可以直接可视化不同框架下网络的模型。Netron支持tf, caffe, keras,mxnet等多种框架模型的可视化，具体地址如下：
> https://github.com/lutzroeder/Netron

Netron安装很简单，具体看官方例子。Netron使用也非常简单，View设置显示内容，点击具体某个层可以看到该层具体参数。通过File-export可是导出网络结构为png图像或者svg图像。Netron具体界面如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809103410955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
## 2.3 ROC曲线绘制
### 2.3.1 二分类ROC曲线绘制
ROC曲线中的主要两个指标就是真正率和假正率，。其中横坐标为假正率（FPR），纵坐标为真正率（TPR)。ROC用于评价模型的预测能力，基于混淆矩阵得出的。TPR越高，同时FPR越低（即ROC曲线越陡），那么模型的性能就越好。曲线下面积AOC（Area Under Curve）被定义为ROC曲线下的面积，使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。
AUC的一般判断标准：
0.5 - 0.7：效果较低，但用于预测股票已经很不错了；
0.7 - 0.85：效果一般；
0.85 - 0.95：效果很好；
0.95 - 1：效果非常好，但一般不太可能。
本文主要通过sklearn.metrics中的roc_curve, auc函数，并通过opencv中的DNN模块调用caffe模型实现分类。二分类ROC曲线绘制python代码(caffe_roc.py)如下：
```python
'''
opencv调用caffe并计算roc
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn import metrics


# 真实图像标签为0的图像路径
imagePath_0 = ['0']
# 真实图像标签为1的图像路径
imagePath_1 = ['1']

# 正类标签
poslabel = 1
# 模型路径
prototxtFile = 'deploy.prototxt'
modelFile = 'model.caffemodel'

# 真实分类结果
trueResult = []
# 检测结果
detectProbs = []

# 图像检测
def detectCaffe(srcImg):
    detectImg = srcImg.copy()
    blob = cv2.dnn.blobFromImage(
        detectImg, 1, (227, 227), (92.713, 106.446, 118.115), swapRB=False)

    net = cv2.dnn.readNetFromCaffe(prototxtFile, modelFile)

    net.setInput(blob)
    detections = net.forward()

    # 分类结果
    order = detections[0].argmax()
    prob = detections[0].max()
    #print('the predict class is:',order)
    #print('the predict class prob is: ', prob)
    # 返回分类结果和概率
    return order, prob

# 图像检测


def imageDetect(detectImagePath, trueLabel):
    for imageFileName in os.listdir(detectImagePath):
        imageFilePath = os.path.join(detectImagePath, imageFileName)
        print("current detect image is: ", imageFileName)
        srcImg = cv2.imread(imageFilePath)
        if srcImg is None:
            print("error image is: ", imageFilePath)
            continue
        detectOrder, detectProb = detectCaffe(srcImg)
        trueResult.append(trueLabel)
        # 如果真实标签和检测结果标签一直直接保存分类概率
        if detectOrder == trueLabel:
            detectProbs.append(detectProb)
        else:
            detectProbs.append(1-detectProb)


# 画ROC图，输入真实标签，模型分类结果，正样本编号
def drawROC(trueResult, detectProbs, poslabel):
    fpr, tpr, thresholds = metrics.roc_curve(
        trueResult, detectProbs, pos_label=poslabel)
    #auc = metrics.roc_auc_score(y, scores)
    roc_auc = metrics.auc(fpr, tpr)

    # 计算约登指数Youden Index（TPR-FPR或者TPR+TNR-1）
    tpr_fpr = list(tpr-fpr)
    bestIndex = tpr_fpr.index(max(tpr_fpr))
    print("约登指数为{}".format(max(tpr_fpr)))
    tprBest = tpr[bestIndex]
    fprBest = fpr[bestIndex]
    thresholdsBest = thresholds[bestIndex]
    print("最佳约登指数阈值为:", thresholdsBest)

    # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # 画出约登指数最大值
    plt.plot(fprBest, tprBest, "ro")
    plt.savefig("roc.png", dpi=300)
    plt.show()

    return fpr, tpr, thresholds, bestIndex

def main():
    # 0标签图像遍历
    for imagePath in imagePath_0:
        imageDetect(imagePath, 0)
    for imagePath in imagePath_1:
        imageDetect(imagePath, 1)
    # poslabel正例标签
    fpr, tpr, thresholds, bestIndex = drawROC(
        trueResult, detectProbs, poslabel)
    np.save('fpr.npy', fpr)
    np.save('tpr.npy', tpr)
    np.save('thresholds', thresholds)
    return fpr, tpr, thresholds


if __name__ == '__main__':
    fpr, tpr, thresholds = main()
```
同时计算约登指数Youden Index（TPR-FPR或者TPR+TNR-1），取使得约登指数最大的阈值为最佳阈值。二分类ROC曲线绘制结果如下图所示，area为AUC值。另外二分类ROC曲线detectProbs 用的是分类概率和真实标签对比绘制ROC曲线。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809181849755.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
### 2.3.2 多分类ROC曲线绘制
由于ROC曲线是针对二分类的情况，对于多分类问题，先将分类标签转换为独热编码。比如n=3时标签转换为：
```python
0->100
1->010
2->001
```
多分类ROC曲线绘制有两种方法：

1. 每种类别下，都可以得到m个测试样本为该类别的概率（矩阵P中的列）。所以，根据概率矩阵P和标签矩阵L中对应的每一列，可以计算出各个阈值下的假正例率（FPR）和真正例率（TPR），从而绘制出一条ROC曲线。这样总共可以绘制出n条ROC曲线。最后对n条ROC曲线取平均，即可得到最终的ROC曲线。
2. 首先，对于一个测试样本：1）标签只由0和1组成，1的位置表明了它的类别（可对应二分类问题中的‘’正’’），0就表示其他类别（‘’负‘’）；2）要是分类器对该测试样本分类正确，则该样本标签中1对应的位置在概率矩阵P中的值是大于0对应的位置的概率值的。基于这两点，将标签矩阵L和概率矩阵P分别按行展开，转置后形成两列，这就得到了一个二分类的结果。所以，此方法经过计算后可以直接得到最终的ROC曲线。

上面的两个方法得到的ROC曲线是不同的，当然曲线下的面积AUC也是不一样的。 在python中，方法1和方法2分别对应sklearn.metrics.roc_auc_score函数中参数average值为'macro'和'micro'的情况。本文主要是应用方法2，方法1太麻烦。方法1可以见：
>https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

caffe下绘制多分类ROC曲线代码(caffe_roc_multi.cpp)如下所示：
```python
'''
opencv调用caffe并计算多分类roc
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn import metrics
from sklearn.preprocessing import label_binarize


# 真实图像标签为0的图像路径
imagePath_0 = ['0']
# 真实图像标签为1的图像路径
imagePath_1 = ['1']
# 真实图像标签2的图像路径
imagePath_2 = ['2']

# 图像分类标签
imageClass = [0, 1, 2]
# 图像分类颜色
classColor = ['aqua', 'darkorange', 'cornflowerblue']

# 模型路径
prototxtFile = 'deploy.prototxt'
modelFile = 'model.caffemodel'

# 真实分类结果
trueResult = []
# 检测分类结果
detectResult = []

# 最佳阈值结果
thresholdsBest = [1]*len(imageClass)


# 图像检测
def detectCaffe(srcImg):
    detectImg = srcImg.copy()
    # 自己输入均值
    blob = cv2.dnn.blobFromImage(
        detectImg, 1, (227, 227), (101.897, 111.704, 121.366), swapRB=False)

    net = cv2.dnn.readNetFromCaffe(prototxtFile, modelFile)

    net.setInput(blob)
    detections = net.forward()

    # 分类结果
    order = detections[0].argmax()
    prob = detections[0].max()
    #print('the predict class is:',order)
    #print('the predict class prob is: ', prob)
    # 返回分类结果和概率
    return order, prob

# 图像检测
def imageDetect(detectImagePath, trueLabel):
    for imageFileName in os.listdir(detectImagePath):
        imageFilePath = os.path.join(detectImagePath, imageFileName)
        print("current detect image is: ", imageFileName)
        srcImg = cv2.imread(imageFilePath)
        if srcImg is None:
            print("error image is: ", imageFilePath)
            continue
        detectOrder, detectProb = detectCaffe(srcImg)
        trueResult.append(trueLabel)
        # 如果真实标签和检测结果标签一直直接保存分类概率
        detectResult.append(detectOrder)


# 画ROC图，输入真实标签，模型分类标签
def drawROC(trueResult, detectResult):
    # 将图像标签二值化
    trueResultBinary = label_binarize(trueResult, classes=imageClass)
    detectResultBinary = label_binarize(detectResult, classes=imageClass)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    # 单独计算每一类ROC值
    for i in range(len(imageClass)):
        # 提取第i类预测数据
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(
            trueResultBinary[:, i], detectResultBinary[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # micro法计算总的roc
    fprMicro, tprMicro, _ = metrics.roc_curve(
        trueResultBinary.ravel(), detectResultBinary.ravel())
    # 计算auc值
    roc_aucMircro = metrics.auc(fprMicro, tprMicro)

    plt.figure()
    # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fprMicro, tprMicro, color='deeppink', label='ROC curve (area = {:0.2f})'.format(roc_aucMircro),
             linestyle=':', linewidth=4)

    # 画出每一类的ROC曲线
    for i in range(len(imageClass)):
        plt.plot(fpr[i], tpr[i], color=classColor[i],
                 label='ROC curve of class{} (area = {:0.2f})'.format(
                     i, roc_auc[i]),
                 )
        # 计算约登指数Youden Index（TPR-FPR或者TPR+TNR-1）
        tpr_fpr = list(tpr[i]-fpr[i])
        bestIndex = tpr_fpr.index(max(tpr_fpr))
        tprBest = tpr[i][bestIndex]
        fprBest = fpr[i][bestIndex]
        thresholdsBest[i] = thresholds[i][bestIndex]
        # 画出约登指数最大值
        plt.plot(fprBest, tprBest, "ro", color=classColor[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="best")

    # 保存图像
    plt.savefig("multi_roc.png", dpi=300)

    return fpr, tpr,trueResultBinary,detectResultBinary


def main():
    # 0标签图像遍历
    for imagePath in imagePath_0:
        if not os.path.isdir(imagePath):
            continue
        imageDetect(imagePath, 0)
    for imagePath in imagePath_1:
        if not os.path.isdir(imagePath):
            continue
        imageDetect(imagePath, 1)
    for imagePath in imagePath_2:
        if not os.path.isdir(imagePath):
            continue
        imageDetect(imagePath, 2)

    fpr, tpr,trueResultBinary,detectResultBinary = drawROC(trueResult, detectResult)
    np.save('fpr.npy', fpr)
    np.save('tpr.npy', tpr)
    np.save('thresholdsBest.npy', thresholdsBest)
    return fpr, tpr,trueResultBinary,detectResultBinary


if __name__ == '__main__':
    fpr, tpr,trueResultBinary,detectResultBinary = main()
```
对于单个分类曲线计算约登指数Youden Index（TPR-FPR或者TPR+TNR-1），取使得约登指数最大的阈值为最佳阈值。多分类ROC曲线绘制结果如下图所示，area为AUC值。另外多分类ROC曲线detectResult 用的是预测标签和真实标签对比绘制ROC曲线。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190809192312664.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
## 2.4 结果可视化部分参考文件
+ [caffe训练日志可视化1](https://blog.csdn.net/u013078356/article/details/51154847)
+ [caffe训练日志可视化2](https://blog.csdn.net/zziahgf/article/details/79215862)
+ [Netscope](http://ethereon.github.io/netscope/#/editor)
+ [Netron](https://github.com/lutzroeder/Netron)
+ [ROC理论1](https://www.jianshu.com/p/82903edb58dc)
+ [ROC理论2](https://www.cnblogs.com/dlml/p/4403482.html)
+ [ROC曲线绘制1](https://blog.csdn.net/xyz1584172808/article/details/81839230)
+ [ROC曲线绘制2](https://blog.csdn.net/YE1215172385/article/details/79443552)
# 3 部署和量化
## 3.1 部署
部署有三种推荐方式，OpenCV DNN模块调用caffe模型/mini-caffe/ncnn。本文主要介绍DNN调用caffe模型和ncnn。
OpenCV调用caffe模型代码(dnn_test.cpp)如下：
```C++
#include "pch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> //dnn模块
#include <time.h>

using namespace std;
using namespace cv;
using namespace ::dnn; //调用DNN命名空间
clock_t start, finish;

String model_file = "model/model.caffemodel"; //模型结构文件
String model_text = "model/deploy.prototxt";  //模型数据

//图像深度学习检测
double detect_NN(Mat detectImg, Net net)
{
	if (net.empty())
	{
		cout << "no model!" << endl;
		return -1;
	}

	//initialize images(输入图像初始化)
	Mat src = detectImg.clone();
	if (src.empty())
	{
		return -1;
	}

	//图像识别转换
	//第一个参数输入图像，第二个参数图像放缩大小，第三个参数输入图像尺寸,第四个参数模型训练图像三个通道RGB的均值（均值文件）
	start = clock();

	Mat inputBlob;

	//resize(src, src, Size(227, 227));
	// 参数分别为输入图像，归一化参数，模型大小，BGR均值
	inputBlob = blobFromImage(src, 1.0, Size(227, 227), Scalar(92.71, 106.44, 118.11));

	Mat prob; //输出结果
		//循环
	for (int i = 0; i < 1; i++)
	{
		net.setInput(inputBlob, "data");
		prob = net.forward("prob"); //输出层2
	}
	Mat probMat = prob.reshape(1, 1); //转化为1行2列
	Point classNumber;				  //最大值的位置
	double classProb;
	//最大值多少
	//最大最小值查找，忽略最小值
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("classidx is:%d\n", classidx);
	printf("prob is %f\n", classProb);
	finish = clock();
	double duration = (double)(finish - start);
	printf("run time is %f ms\n", duration);
	return duration;
}

int main()
{
	Net net = readNetFromCaffe(model_text, model_file);
	Mat detectImg = imread("image/cat.jpg");
	double runTime = detect_NN(detectImg, net);
	return 0;
}
```
以上只是caffe简单调用分类模型。其他网络调用可以参见以下链接：
>https://docs.opencv.org/4.1.1/d6/d0f/group__dnn.html
>https://blog.csdn.net/LuohenYJ/column/info/34751

另外一种就是通过ncnn，ncnn调用已经写过博客，可以参见以下链接：
>https://github.com/Tencent/ncnn/wiki/ncnn-%E7%BB%84%E4%BB%B6%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97-alexnet
>https://blog.csdn.net/LuohenYJ/article/details/97031156

实际ncnn使用参考ncnn在github/examples目录文件调用。地址如下：
https://github.com/Tencent/ncnn/tree/master/examples
但是不管ncnn都需要输入模型均值，输入图像后要减去训练均值。如果不知道均值，只有binaryproto文件，可以建立mean.py读取binaryproto文件里面的信息,获得BGR均值。代码如下：
```python
#coding=utf-8
import caffe
import numpy as np

# 待转换的pb格式图像均值文件路径
MEAN_PROTO_PATH = 'mean.binaryproto'
# 转换后的numpy格式图像均值文件路径
MEAN_NPY_PATH = 'mean.npy'

# 创建protobuf blob
blob = caffe.proto.caffe_pb2.BlobProto()           
# 读入mean.binaryproto文件内容
data = open(MEAN_PROTO_PATH, 'rb' ).read()         
# 解析文件内容到blob
blob.ParseFromString(data)

# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
array = np.array(caffe.io.blobproto_to_array(blob))
# 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
mean_npy = array[0]
print(mean_npy.mean(1).mean(1))
np.save(MEAN_NPY_PATH ,mean_npy)
```
## 3.2 量化
caffe模型太大，速度太慢。可通过float32转int8进行模型压缩和速度提升。量化的意思就是一般而言，神经网络模型的参数都是用的32bit长度的浮点型数表示，实际上不需要保留那么高的精度，可以通过量化，比如用0~255表示原来32个bit所表示的精度，通过牺牲精度来降低每一个权值所需要占用的空间。常用量化方法为int8量化，float 32进行int 8量化，能够使模型尺寸更小、推断更快、耗电更低。唯一的缺点，模型精度会下降。
int8量化主要方法可以见下图(图来自网上):
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190810100516419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
本文主要在ncnn下实现caffe的int8量化。caffe的int8量化项目地址为：
>https://github.com/BUG1989/caffe-int8-convert-tools

量化方法非常简单,首先
```python
git clone https://github.com/BUG1989/caffe-int8-convert-tools
```
然后进入caffe-int8-convert-tools，把验证集和caffe模型参数放入该目录，输入以下命令实现量化：
```ptyhon
python caffe-int8-convert-tool-dev-weight.py --proto=model/deploy.prototxt --model=model/deploy.caffemodel --mean 92.713 106.446 118.15 --images=val/ --output=model.table
```
官方教程命令为：
```python
python caffe-int8-convert-tool-dev-weight.py --proto=test/models/mobilenet_v1.prototxt --model=test/models/mobilenet_v1.caffemodel --mean 103.94 116.78 123.68 --norm=0.017 --images=test/images/ output=mobilenet_v1.table --group=1 --gpu=1
```
使用教程如下：
```python
$ python caffe-int8-convert-tool.py --help
usage: caffe-int8-convert-tool.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--gpu GPU]

find the pretrained caffe models int8 quantize scale value

optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained weights
  --mean MEAN           value of mean
  --norm NORM           value of normalize
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --gpu GPU             use gpu to forward
```
proto/model是caffe模型参数文件，mean为BGR均值，norm是caffe模型归一化参数，有些模型没有归一化就不要填，image是用于校准图像目录(jpg图像)。量化时需要校准，具体原理见:
>https://arleyzhang.github.io/articles/923e2c40/

这样量化后获得输出文件model.table，在将caff模型转换为ncnn模型时，加入model.table，如下所示，就可以得到量化的模型。
>./caffe2ncnn deploy.prototxt model.caffemodel model_int8.param model_int8.bin 256 model.table

上面256指的是quantizelevel量化级别，如果为0就不进行量化。如果某些层不想量化，打开model.table，输出那一层的量化参数。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190810110419375.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0x1b2hlbllK,size_16,color_FFFFFF,t_70)
调用int量化后的模型，和原来ncnn调用不量化的模型一样，ncnn会自动调用量化模型。
## 3.4 部署和量化部分参考文件
+ [OpenCV调用Caffe1](https://blog.csdn.net/shanglianlm/article/details/80030569)
+ [OpenCV调用Caffe2](https://www.jianshu.com/p/fdf9c3b70dd4)
+ [OpenCV深度学习调用实例](https://github.com/opencv/opencv/tree/master/samples/dnn)
+ [ncnn入门1](https://blog.csdn.net/weixin_45250844/article/details/94910897)
+ [ncnn入门2](https://blog.csdn.net/qq_36982160/article/details/79929869)
+ [caffe量化1](https://blog.csdn.net/u014644466/article/details/83278954)
+ [caffe量化2](https://blog.csdn.net/qq_33431368/article/details/85029041)
+ [ncnn量化详解](https://zhuanlan.zhihu.com/p/71881443)

# 4 总结
本文只写了caffe分类模型的相关笔记，目标检测就没写了。因为用darknet的yolo比较多。有空在写caffe目标检测的笔记。
