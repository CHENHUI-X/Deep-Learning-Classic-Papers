# R-CNN 
## R Girshick · 2013
### arXiv:1311.2524
- **使用selective search 在原image上搜寻大约2K个region**
- ***然后将这些region丢入到CNN中抽取特征***
- 最后接上两个fully connect layer
	- 将提取到的特征存起来（save in disk，which is slowly)
- 最后followed by K binary SVM classifier .
	- 使用存储起来的特征，训练SVM分类器

***使用SS抽取region与后续的操作是分开的。***
***
***
# SPP-net
## K He · 2014 
### arXiv:1406.4729
- **使用selectivesearch 在原image上搜寻大约2K个region**
- 只不过现在是***只把原image丢入到CNN中抽特征***
- 因为使用filter在image上卷积，通过padding，可以实现卷积不改变大小，然后通过poolling实现尺寸减半，那么最后的feature map的位置是和原来image位置是一一对应的。那么原来在***原图上搜到的region就能直接从feature map位置对应上***
- 然后使用 ***spatial pyramid pooling layer***在对应位置上抽取特征
- 最后接上两个fully connect layer
	- 将提取到的特征存起来（save in disk，which is slowly)
- 最后followed by K binary SVM classifier .
	- 使用存储起来的特征，训练SVM分类器

***使用SS抽取region与后续的操作是分开的。***
***
***
# Fast R-CNN
## R Girshick · 2015
### arXiv:1504.08083
- ***使用selectivesearch 在原image上搜寻大约2K个region***
- 只不过现在是***只把原image丢入到CNN中抽特征***
- 在feature map 上对应region位置（原image位置的1/32处）抽取特征，只不过***这里使用与SPP-net不同的 spatial pyramid pooling*** ，而是使用了***ROL poolling layer***，这个layer是spp layer的一个简化版，spp是使用不同scale的poolling size，这里使用***固定size的 spatial poolling size***。
- 接上两个fully connect layer，拉成一维向量
- *然后接下来就是一个比较新的地方*
	- fast R-CNN 将抽到的特征直接丢进 ***两个并列的layer***，一个负责输出这个region的class，一个进行regression输出位置信息。
- ***首创了classification loss 和 regression loss 一起训练*** ： ***loss = loss for class + loss for regression***

***使用SS抽取region与后续的操作是分开的。***
***
***
# Faster R-CNN
## S Ren · 2015
### arXiv:1506.01497
- 提出了一种 region proposal net work 来生成 region
	- ***抛弃了传统的SS method获取region***
- 该 RPN 首先吃一整张image，然后使用CNN抽出feature  map 
- 接着 对 FM 的每一个cell（位置），创建K个anchor box（原文使用3个框，3中不同尺寸）
- 在每个cell使用3 * 3 * 256 抽出这个cell的特征
- 再***并列的使用两个CNN***，一个是1 * 1 * 2 * K ，输出这个cell的K个box 是/否含有obj的概率
- 另一个使用1 * 1 * 4 * K 的filter 输出这个cell上K个box的位置
	- ***可以看到，训练RPN产生anchor box 时抛弃了fully connect layer***
-  接下来train这个RPN网络，这个网络使用
***loss = loss for class（binary） + loss for regression***以产生anchor box
- 这个 **RPN 训练好之后**（能够产生很好的anchor box or called region like previous work ，only differerence is the anchor box   on the feature map ）
- 然后***再另外训练一个新的detection work（使用 Fast R-CNN）***，只不过这个 net ***训练用的region***（anchor box） 位置大小信息***使用的是上一个RPN产生的***。
- 等到现在***这个detection work训练好之后***，再用这个训练好的detecter ***把之前的RPN网络参数覆盖掉***（conv layer），然后***只 fine tune the layer unique to RPN***（***再训练RPN***，产生更好的或者更合适的anchor box）。
	- 这样***实现了两个网络的conv layer 是一样的***。
- 然后***将最新得到的anchor box 和 image 带入 现有的 detection net***（一个fast R-CNN），***再次训练***，只不过固定CNN层，就***只更新fc部分***。
	- 上述过程类似交替迭代更新

***抽取region与后续的操作是分开的。***
***只不过用了net work 来识别region or  anchor box***
***在detection的时候，还是使用了FC层***
***
***
# YOLO - V1
## J Redmon · 2015
###  arXiv:1506.02640
- 将原image划分为7×7个格子，每个格子上预测2个框
- 直接对原image使用CNN抽特征
- 最后接两个fully connect layer 直接输出 7*7*（5+5+20）
	- 其中 5表示框的位置信息（x,y,w,h)和conf，20表示直接输出具有最大conf且框中包含obj的那个框所含obj类别概率。
- 需要注意的是，预训练的时候用的224的size训练的，之后把初始化好的net用到detection的时候，用的是448的size训练的（这会导致net不太适应，V2解决了这个问题）

***开创了一步到位的训练模式***，之前都是需要先用ss获取region，然后进行后续的操作。
***
***
# SSD: Single Shot MultiBox Detector
## Wei Liu
### arXiv:1512.02325
![SSD.png](https://github.com/CHENHUI-X/Deep-Learning-Classic-Papers/blob/master/img/SSD.png)
这片文章一个比较突出的贡献是，直接***在不同resolutions的feature map*** 上的***每一个cell*** 直接使用 ***3 * 3 *  (K * (C +  4 )) 的 filter输出 vector or feature*** ,where ***C is the score of c class , 4 is the offset of this cell , K is number of box/anchor box on  this  resolutions***

***贡献就是使用不同 resolution 的 FM 抽特征，实现了不同分辨率的特征抽取，能够对小部件检测友好***
***
***
# YOLO9000  or YOLO -V2
## Joseph Redmon
###  arXiv:1612.08242 
yolo - v2 主要做了以下工作
- batch normalization 
	- 使用了 ***batch normalization***
- high resolution classifier 
	- 由于之前yolo -v1初始化net的时候，是使用imagenet数据集（224 * 224）上预训练的网络，而在训练detection的时候，用的又是448 * 448 训练的，会导致模型不太适应。因此V2采取的策略是***先用224 * 224 的***ImageNet dataset 训练***160个epoch***，然后再fine tune the ***size*** of image in imagenet dataset ***to 448 * 448 ，再把net work 跑10个epoch***。最后考虑到最后一层的***feature map 是原image size 的1/32***，***想让FM是奇数个长宽cell***（方便一个obj只有一个center），所以这里最后实际在训练detection的时候，***输入的image size 是 416 * 416，输出FM为 13 * 13*** 的
- darknet
	- 使用了一种***新的CNN架构-darknet***（当然还是下采样32倍）
- dimension priors
	- 之前anchor box 要么是使用ss得到，要么是人为规定个数和尺寸（	Faster R-CNN 3个框，3种纵横比，共9个；SSD则指定不同level FM 上 anchor box 的size 不一样），有没有一种方法得到更加客观或者少而精的box？本文使用了***K均值***，在训练集上，对所有image中obj对应的框的位置（相对）和大小（相对）进行聚类，***发现取5个框（得到框的大小）*** 的时候能够取得一个trade-off。于是train的时候在每个cell上用这5个box的基础上去train。
- location predict 
	- use  ***sigmiod function to tune the offset to range(0，1)***
- passthrough ： ***不同level的FM 连接到一起***
	- use ***second last layer feature  map 26 * 26 * 512 -> 13 * 13 * 2048*** ,than ***concatenate  13* 13* 2048 and 13 * 13 * 1024***,which come  from  last conv layer 
- mutu - scale 
	-	 ***Every 10 batches our network randomly chooses a new image dimension size***
-  high resolution detecter	
	-  when the net is ***used  for detection*** , fine tune the net ,***by removing the last convolutional layer and instead adding on three 3 × 3 convolutional layers with 1024 filters each followed by a final 1 × 1 * N*** convolution，which is used  to produce the output,that is N  = K*(5 + num of  class) 
- 最后train的时候，把class image and detection image mixed，***when meet class image，only update loss that part of net class，when meet detection image，update all loss。***
- Hierarchical classification
	- 因为这里把image for class 和 image for detection mixed了，所以他们是类别不一样的，class多，detection的比较少，所以这里使用了一个技巧，但是我没怎么看。。。。。。

***没有什么大的更新，都是一些技巧，需要注意的是，这里不再使用任何 fully connect layer***
***
***
# YOLO -V3
## Joseph Redmon
###  arXiv:1804.02767
[implement for yolov3](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
yolo -v3 主要做了几个新的改进
- 改进了darknet work ，使用了残差连接，以及将不同level的feature map 通过 upsample 然后叠加起来
- 也使用k均值收集框的大小，而且看代码的话，是不同尺寸上的feature map ，使用不同大小的框，并不是9个框在同一个level的feature map上同时使用。
- 其他没什么大的改动了。

[yolo-v3架构](https://blog.csdn.net/qq_37541097/article/details/81214953)
![yolo-v3架构.jpg](../_resources/yolo-v3架构.jpg)

