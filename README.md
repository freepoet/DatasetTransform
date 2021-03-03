### 2021.03.03
- **DOTA数据集格式转成VOC格式**
 
 DOTA2VOC

### 2021.03.01
- **绘制符合北航学报的曲线图**
 analyze_logs_buaa.py

### 2021.02.25
- **SSDD目标尺寸分布**
D20210225

### 2021.01.23
- **在fork库上进行push是不计算contribution的**

### 2021.01.18
- **撤销commit**

  [撤回commit操作](https://blog.csdn.net/w958796636/article/details/53611133)

  git reset --soft HEAD^.
 
  仅仅是撤回commit操作，您写的代码仍然保留，HEAD^的意思是上一个版本，也可以写成'HEAD~1'
  
  如果你进行了2次commit，想都撤回，可以使用HEAD~2。

---

## 《深度学习之PyTorch物体检测实战》代码
## 2020/12/09
__前向传播__</br>
test01.py</br>
__两层感知机模型(全链接)__</br>
test02.py  perception.py
## 2020/12/10
__交叉熵损失__</br>
score和label的size是不同的，具体参考陈云的书P116</br>
test01.py</br>
__梯度回传__</br>
test02.py</br>
__模型加载__</br>
test03.py</br> 
__tensorboard使用(有问题)__</br>
test04.py</br>
__激活函数__</br>
test05.py</br>  
## 2020/12/11
__测试VGGNet__</br>
vgg_test.py</br>
__测试Inceptionv1__</br>
inceptionv1_test.py  
__测试Inceptionv2__</br>
inceptionv2_test.py  
__测试ResNet__</br>
resnet_bottleneck_test.py</br>
__测试DenseNet__</br>
densenet_block_test.py</br> 
## 2020/12/12
__特征金字塔fpn__(没看太懂）</br>
fpn.py</br>
__测试fpn__</br>
fpn_test.py 
## 2020/12/13
__测试detnet的bottleneck__</br>
detnet</br>
# 《深度学习框架Pytorch入门与实践》代码
## 2020/12/14
__Faster RCNN实现__</br>
faster-rcnn-pytorch</br>
__Tensor基本运算__</br>
test221.py</br>
__Autograd自动微分__</br>
test222.py</br>
__LeNet5__</br>
test223.py</br>
只要在nn.Module的子类中定义了forward函数，backward就会自动实现
## 2020/12/16
__LeNet训练CIFAR10__</br>
test224.py</br>
数据集加载->定义网络->定义损失函数/优化器->模型训练(输入数据+梯度清零+forward/backward+更新参数)->模型测试
## 2020/12/17
__Tensor基本操作__</br>
test311.py</br>
__Tensor与Numpy转换__</br>
test312.py</br>
__Tensor内部结构__</br>
test313.py</br>
__torch的存储__</br>
test314.py</br>
__plt绘图__</br>
test315.py</br>
## 2020/12/18
__Variable基本操作__</br>
test321.py</br>
__计算图__</br>
test322.py</br>
__Variable autograd实现linear regression__</br>
test324.py</br>
__图片处理__</br>
test421.py</br>
__交叉熵损失函数__</br>
test422.py</br>
## 2020/12/19
__图片数据处理__</br>
test51.py</br>
## 2020/12/20
__计算机视觉工具包torchvision__</br>
test52.py</br>
问题：doesn't match</br>
__Tensorboard的使用__</br>
test531.py</br>
问题：tensorflow环境下notebook无法打开</br>
__Visdom的使用__</br>
test532.py</br>
在两次尝试tensorboard失败后，改用visdom</br>
在pytorch131环境下输入python -m visdom.server启动visdom</br>
问题：一个图中绘制多条曲线</br>
__使用GPU加速：cuda__</br>
test54.py</br>
## 2020/12/21
__DogCat分类__</br>
D20201221</br>
若要可视化，则要先打开visdom，并刷新</br>
## 2020/12/22
__ipdb进行debug__</br>
test621.py</br>
## 2020/12/24
尝试改写chenyun的[基于pytorch的FasterRCNN简单实现](https://github.com/freepoet/simple-faster-rcnn-pytorch)，似乎不能trainning from scartch,</br>
改用[mmdeteciton](https://github.com/freepoet/mmdetection)</br>
```python
python tools/train.py configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py --work-dir output1224  --gpus 1
```
最原始faster_rcnn训练SSDD数据集</br>
## 2020/12/26
__熟悉hook机制__</br>
hook_test.py</br>
__初始化策略__</br>
test45.py</br>
__nn.Module深入分析__</br>
test46.py</br>
问题： x=self.param1@input
## 2020/12/28
__理解董洪义的书本P115关于FasterRCNN的代码__</br>
__np.vstack()、np.hstack()、zip()的使用__</br>
test1.py</br>
## 2020/12/29
__在mmdetection里面修改FasterRCNN网络结构__</br>
## 2020/12/30
__NMS__</br>
test1.py</br>
__items()用法__</br>
test1.py</br>
## 2021/01/04
__DTCWT与最大池化对比__</br>
test01.py</br>
注意VOC的JPEGImages文件夹下的图片4被更改了</br>
__DWT、图片频谱分析__</br>
test02.py</br>
## 2021/01/08
__Faster-RCNN结合Wavelet Pooling结合__</br>
单GPU训练</br>
``` python
python tools/train.py configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py --work-dir output20210108 --gpus 1
```
多GPU训练（wavelet包似乎不支持）</br>
``` python
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py 4
```
__图片相似度__</br>
test01</br>
## 2021/01/09
__图片5种相似度度量方法__</br>
test01.py</br>
## 2021/01/10
__根据训练结果json文件绘制曲线__</br>
[mmdetection给出的方法](https://github.com/open-mmlab/mmdetection/blob/c551b598bb5c5ddea201b4a451341264d4a8f26b/docs/useful_tools.md)</br>
test01.py</br>
## 2021/01/11
__pywt小波变换包的使用__</br>
仅仅支持cpu</br>
## 2021/01/14
__mmdetection中自定义Wavelet CNN__</br>
## 2021/01/15
__图片相似性的度量，之间的方法有误，应该是整幅图片大小不变的前提下，目标尺寸进行缩放__</br>
__PIL matplot cv2读取显示图片三种方式的比较__</br>
test01.ipynb</br>
__xml标注文件可视化__</br>
test02.py</br>
[mmdetection中的方法:show_result()](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/base.py)</br>
## 2021/01/17
### mmdetection:ResNet50不用FPN
``` python
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco.py --work-dir output2021011701  --gpus 1
```
需要将**configs/_base_/datasets/coco_detection.py**里的samples_per_gpu改为1，否则会报错：out of memory</br>
out_indice=(2,)  使用fpn的话out_indice=(0,1,2,3)  outs只有一个元素，相当于只取了layer3的输出,size：N C H/4 W/4</br>
**two_stage.py**里的x = self.backbone(img)输出x也只有一个元素</br>
由于不使用FPN,所以self.with_neck=False</br>
### mmdetection:ResNet50不用FPN并且training from scratch(without pretrained model)
将faster_rcnn_r50_cafffe_c4.py中的pretrained改为None
### mmdetection:ResNet101使用FPN并且training from scratch(without pretrained model)
将faster_rcnn_r101_fpn_1x_coco.py中的pretrained改为None
