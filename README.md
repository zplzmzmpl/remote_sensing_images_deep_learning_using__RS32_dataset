# remote_sensing_images_deep_learning_using__RS32_dataset
*rs images deep learning classification, use dataset RS32(collected by WHU)*

**数据集说明如下:**

HelloRS数据集由12800张32*32的 RGB 彩色图片构成，共10个分类，分别为：水域（Waters）、森林（Forest）、农用地（CultivateLand）、河流（River）、高速路（Highway）、高压线塔（Pylon）、游泳池（SwimmingPool）、网球场（TennisCourt）、篮球场（BasketballCourt）、足球场（FootballField）。采用按分类目录存放数据，包括验证集、训练集、测试集等，其结构为与内容如下：

..../HelloRS32/

    -------/All/---------imgs
    
    -------/train/-------imgs
    
    -------/train_val/---imgs
    
    -------/test/--------imgs
    
    -------/val/---------imgs

<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/9cad7115-2c64-4d90-aafa-6d84b85b9b12"></div>

## 1 基于SVM和Softmax损失函数的线性分类器
### 1.1 使用线性分类器。
对于图像分类而言，线性分类器计算图像中3个颜色通道中所有像素的值与权重的矩阵乘，从而得到分类分值。线性分类器：从最简单的概率函数开始，一个线性映射： f(xi,W,b)=Wxi+b，假设每个图像数据都被拉长为一个长度为D的列向量，大小为[D x 1]。其中大小为[K x D]的矩阵W和大小为[K x 1]列向量b为该函数的参数（parameters）。参数W被称为权重（weights）。b被称为偏差向量（bias vector），这是因为它影响输出数值，但是并不和原始数据产生关联。
### 1.2 预处理步骤
所有图像都是使用的原始像素值（从0到255）。但在机器学习中，对于输入的特征做归一化（normalization）处理是常见的套路。而在图像分类的例子中，图像上的每个像素可以看做一个特征。在实践中，对每个特征减去平均值来中心化数据非常重要。该步骤意味着根据训练集中所有的图像计算出一个平均图像值，然后每个图像都减去这个平均值，零均值的中心化是很重要的。
### 1.3 计算梯度的两种方法
一个是缓慢的近似方法（数值梯度法），但实现相对简单。另一个方法（分析梯度法）计算迅速，结果精确，但是实现时容易出错，且需要使用微分。使用有限差值近似计算梯度比较简单，但缺点在于终究只是近似（因为我们对于h值是选取了一个很小的数值，但真正的梯度定义中h趋向0的极限），且耗费计算资源太多。第二个梯度计算方法是利用微分来分析，能得到计算梯度的公式（不是近似），用公式计算梯度速度很快，唯一不好的就是实现的时候容易出错。为了解决这个问题，在实际操作时常常将分析梯度法的结果和数值梯度法的结果作比较，以此来检查其实现的正确性，这个步骤叫做梯度检查。
### 1.4 损失函数（Loss Function）（有时也叫代价函数Cost Function或目标函数Objective）
衡量我们对结果的不满意程度。直观地讲，当评分函数输出结果与真实结果之间差异越大，损失函数输出越大，反之越小。
### 1.5 随机梯度下降Stochastic Gradient Descent （SGD）
随机梯度下降是mini-batch梯度下降的一个特例，每次使用的mini-batch只包括一个样本的情况就是SGD。但是在实际应用中比较少见随机梯度下降，因为一次计算多个样本梯度会更加高效。实际中的SGD通常被用来指代mini-bach，其中，Mini-batch的大小也是一个超参数，但是它不需要使用交叉验证选择；而是通常根据内存大小来制定，常常使用2的幂：32、64、128，这样更容易实现向量化操作，且运算更快。
### 1.6 实现流程图

<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/5acaf696-99b8-4958-87e7-ebfe46d8eb01"></div>

### 1.7 结果验证

<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/cb4e3684-3157-432f-924b-9a32a66913b5"></div>
<div align=center>SVM损失函数</div>
<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/4bbbeead-4382-40fc-b6c5-da1735034495"></div>
<div align=center>Softmax损失函数</div>

SVM训练时间为26s，训练精度为0.14309，softmax训练时间为43s，训练精度为0.448568 可知，softmax训练时间大致为SVM的2倍，但训练精度为SVM的3倍所以此处softmax训练效果明显优于SVM。
SVM和Softmax经常是相似的：通常说来，两种分类器的表现差别很小。对本次实习结果而言softmax训练效果明显优于SVM。相对于Softmax分类器，SVM更加“局部目标化（local objective）”，这既可以看做是一个特性，也可以看做是一个劣势。对于softmax分类器，情况则不同。softmax分类器对于分数是永远不会满意的：正确分类总能得到更高的可能性，错误分类总能得到更低的可能性，损失值总是能够更小。但是，SVM只要边界值被满足了就满意了，不会超过限制去细微地操作具体分数。这可以被看做是SVM的一种特性。

### 1.8 超参数调试
调整学习率，正则化。学习率越大, 输出误差对参数的影响就越大, 参数更新的就越快, 但同时受到异常数据的影响也就越大, 很容易发散。学习率 (Learning rate，η) 作为监督学习以及深度学习中重要的超参，其决定着目标函数能否收敛到局部最小值以及何时收敛到最小值。合适的学习率能够使目标函数在合适的时间内收敛到局部最小值。

运用梯度下降算法进行优化时，权重的更新规则中，在梯度项前会乘以一个系数，这个系数就叫学习速率 α。学习率如果过大，可能会使损失函数直接越过全局最优点，容易发生梯度爆炸，loss 振动幅度较大，模型难以收敛。学习率如果过小，损失函数的变化速度很慢，容易过拟合。会大大增加网络的收敛复杂度. 虽然使用低学习率可以确保我们不会错过任何局部极小值，但也意味着我们将花费更长的时间来进行收敛，特别是在被困在局部最优点的时候。

正则化在机器学习中起到了一种正则化参数的作用，它可以帮助减少模型的过拟合问题。过拟合是指模型在训练集上表现很好，但在未知数据上表现较差的情况。通过在损失函数中引入一个正则化项，来限制模型的复杂度。常用的正则化方法有L1正则化和L2正则化。L1正则化通过在损失函数中加上参数的绝对值之和的乘以一个正则化系数来实现。L1正则化可以使得部分参数变为0，从而达到特征选择和稀疏性的效果。本次实习使用L2正则化，通过在损失函数中加上参数的平方和的乘以一个正则化系数来实现。L2正则化可以使得参数值尽量小，从而降低模型的复杂度。

<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/b9880d0c-182b-4a9a-b534-5ee1650d6559"></div>
<div align=center>SVM与Softmax超参数调试</div>

softmax分类器精度高于SVM，当学习率设置为1e-7以及正则化设置为3e4时取得最佳精度为0.479823.

### 1.9 权重可视化
由下图对比可知，softmax分类输出权重特征相较SVM输出权重更加突出，因此再次证明本次实习中softmax分类器更优。
<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/4f6ebf44-d8c7-456a-b4a5-acef74f65765"></div>

### 1.10 测试
在测试集上使用最佳模型分类，并抽样展示测试集中对每个类别的图像样本，正确预测为蓝色字体，错误预测样本为红色字体。由下图结果可知，模型对水体、篮球场、足球场的分类错误率高，水体与森林混淆率高，篮球场与网球场混淆率高，足球场和网球场、森林、游泳池混淆率高。
<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/adc4d0c1-fea1-4966-84e0-3763d1f606e5"></div>
交叉验证时在测试集上，SVM得分为0.387139，softmax得分为0.479823 抽样验证时，由上两个单元可知，标题红色字体为错分，蓝色字体为正确分类，SVM分类精度为0.35714285714285715，softmax分类精度为
0.44285714285714284 在相同测试集上，交叉验证与抽样二者精度大致相符，SVM为(0.387139-0.35714285714285715)/0.387139=77.4%,softmax为(0.479823-0.44285714285714284)/0.479823=77.04%

## 2 实现简单的全连接层网络
### 2.1 前向传播（forward）。
简单理解就是将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止。
### 2.2 反向传播（backward）
是利用链式法则递归计算表达式的梯度的方法。理解反向传播过程对于理解、实现、设计和调试神经网络非常关键。微积分中的链式法则（为了不与概率中的链式法则相混淆）用于计复合函数的导数。反向传播是一种计算链式法则的算法，使用高效的特定运输顺序。

*下图为本次两层网络结构，前向affine1仿射层非线性变换-->ReLu1激活函数-->affine2Relu-->softmax-->反向ReLu2梯度-->affine2梯度-->ReLu1梯度-->affine1梯度。*
<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/0241e4e2-a57e-427e-8b50-06d4e22b1782"></div>

### 2.3 实现流程图
<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/a705f06a-1f0b-4d60-8ae4-08e953385f3d"></div>

### 2.4 训练结果
<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/f75e1179-ecad-4bff-96fc-f3ab55b7898c"></div>
<div align=center>SGD训练结果</div>
使用梯度下降法SGD训练网络，梯度是一个矢量，它告诉我们权重的方向。更准确地说，它告诉我们如何改变权重，使损失变化最快。我们称这个过程为梯度下降，因为它使用梯度使损失曲线下降到最小值。随机的意思是“由偶然决定的”。我们的训练是随机的，因为小批量是数据集中的随机样本，因此称为SGD。 优化时使用指数学习率时间表来调整学习速率; 在每个epoch之后，通过将学习速率乘以衰减速率来降低学习速率。得到最终验证精度为0.72666

损失函数、训练精度及验证精度可视化结果如下图所示，它在曲线上的特点是training acc和validation acc都已经收敛并且之间相差很小很小。如下图所示，模型在50轮过后，两个acc曲线都开始收敛，而且两者之间并没有肉眼的差距。 通常traing acc会更大，这样他们之间就会有个gap，这个gap叫做generalization gap。可知曲线符合要求：

<div align=center><img src="https://github.com/zplzmzmpl/remote_sensing_images_deep_learning_using__RS32_dataset/assets/121420991/ac51f405-63d1-4700-bfb3-969c4987ba60"></div>
<div align=center>损失曲线与训练、验证曲线</div>

### 2.5 网络优化
