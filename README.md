# Vitrual Try On Flask

虚拟试衣网络的核心代码仓库。 

实现功能:  
 - 实现快速的人体图和衣服图到穿搭图片的合成
 - 基于Flask提供Web服务和后端响应服务
 - 基于SKU提供推荐穿搭功能
 - 基于安卓提供多平台客户端

启动：`python main.py`

如果你只是想使用纯网络模型，请见`Model.py`和模板[notebook](http://github.com/GrayXu/Virtual-Try-On/blob/master/Template.ipynb)  

其中核心的外部调用接口为[predict](http://github.com/GrayXu/Virtual-Try-On/blob/master/Model.py#L90)函数，关注文档注释提及的5个flag参数即可。
```
def predict(self, human_img, c_img, need_pre=True, need_bright=False, keep_back=False, need_dilate=False, check_dirty=False):
    '''
    输入：
        human_img为人体图片，c_img为衣服图片,均为numpy array with shape(256,192,3) RGB
    五个flag:
        need_pre为是否需要预处理（crop+resize到256*192），在LIP数据集上关闭need_pre效果较好（upsample损失）
        need bright为亮度增强，
        keep back为基于mask保持除了上衣以外的部分，
        need dilate为膨胀柔化keep back的mask，需要keep_back开启 
        check_dirty为增加检查数据是否有肢体遮挡交叉（可能检测不全）  
    返回：
        一个(256,192,3)的穿上衣服的图片，
        画面中人体的置信度，0则为触发强排除条件（检测难样本等）
    '''
    ...
```


## 依赖

torch==1.2.0  
tensorflow==1.12.0  
torchvision==0.2.0  

目前代码需要在GPU环境运行，若要修改到CPU环境，请删除所有CPVTON.py和Networks.py两个文件中所有的`.cuda()`调用。

## 占用

显存约占用7.12GB，单张图片预测约为0.62s（JPPNet 0.6s, CPVTON 0.02s）。  
*若关注Real Time，请更换人体特征提取思路，如尝试使用CE2P*

## 文件说明

文件名 | 功能  
-|-  
main.py | flask服务相关  
get.py | 向flask发起predict请求  
Model.py | 提供虚拟试衣网络的外部接口
CPVTON.py | CPVTON模型整合预测代码
networks.py | CPVTON模型网络定义
JPPNet.py | JPP-Net 预测代码
static/* | flask的静态访问资源
data/\*, example/\* | 样例测试图片
template/\* | flask的模板html based on Jinjia
checkpoints/\* | 存预训练模型
match_server* | 提供衣服推荐接口相关逻辑

## checkpoints下载

*coming soon..*

## 参考

模型的设计基于JPP-Net与CPVTON，及其开源代码。  

[**VITON**: An Image-based Virtual Try-on Network](https://arxiv.org/abs/1711.08447v1),Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, Larry S. Davis. **CVPR 2018**

[(**CP-VTON**) Toward Characteristic-Preserving Image-based Virtual Try-On Networks](https://arxiv.org/abs/1807.07688), Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, Meng Yang. **ECCV 2018**

[(**JPP-Net**)Look into Person: Joint Body Parsing & Pose Estimation Network and A New Benchmark](https://arxiv.org/abs/1804.01984), Xiaodan Liang, Ke Gong, Xiaohui Shen, Liang Lin, **T-PAMI 2018**.

----

*Powered By **Imba**, [JD Digits](https://www.jddglobal.com/)*