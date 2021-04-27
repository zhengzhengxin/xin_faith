### 文件须知
#### 1、cfg.py：配置了大量参数







### 写这个代码学习的东西
#### 1、动态调整学习率的方法
```python
#在指定的epoch值，如[10,15，25，30]处对学习率进行衰减，lr = lr * gamma
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,25,30], gamma=0.1)
```
#### 2、函数形参使用**
*args 和 **kwargs 主要用于函数定义。

你可以将不定数量的参数传递给一个函数。不定的意思是：预先并不知道, 函数使用者会传递多少个参数给你, 所以在这个场景下使用这两个关键字

*args 表示任何多个无名参数，它本质是一个 tuple
**kwargs 表示关键字参数，它本质上是一个 dict

#### 3、一些函数的作用
- torch.from_numpy(ndarray) → Tensor，即 从numpy.ndarray创建一个张量
- extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
- numpy.nan_to_num(x):
使用0代替数组x中的nan元素
- torch.randn 方法
返回一个张量，其中包含来自均值为0、方差为1的正态分布的随机数(也称为标准正态分布)