---
layout: post
title: "线性回归模型"
subtitle: '机器学习基础篇一'
author: "Arthur"

header-style: text
tags:
  - Machine Learning
---


基础模型回顾学习系列

[TOC]

简介
---
线性回归是数理统计中，利用数理统计中回归分析，来==确定两种或两种以上变量间相互依赖的定量关系==的一种统计分析方法  
包括一元线性回归（一个自变量）和多元线性回归(多个自变量)


原理推导
-------
- 问题建模  
就举经典的例子
> 有一组房屋数据
> 两个维度：房屋面积(X)与售价(Y)  
> 找出二者之间的关系  

![image](http://images.cnblogs.com/cnblogs_com/jerrylead/201103/201103052209058937.png)  

需要找出一条直线来拟合二者的关系

```math
y=Wx+b  

```
x:房屋面积；y:价格；如果有多个维度（多元回归）：  
```math
h(x)={\beta_1}x_1+{\beta_2}x_2+{{\beta_3}x_3}+...+b  
  
  =>  
 
  h(x)={\beta}^TX (x_0=1)
```
对于多元回归，使用矩阵形式进行表示
```math
h_{\beta}(X)=
\left(               
  \begin{array}{ccc}   
  \beta1 & \beta2 & \beta3\\ 
  \end{array}
\right)               
*
\left(               
  \begin{array}{ccc}   
  x1\\
  x2\\
  x3\\
  \end{array}
\right)
```
- 求解过程
模型求解涉及具体的决策和算法  
决策即为：损失函数/风险函数  
算法即为：具体求解参数的方法  
损失函数是用来评估模型建模的效果  
描述h函数不好的程度  
使用模型函数值即估计值与实际值的差的平方和作为损失函数 1/2是用于在求导中消除前面的参数  
==**损失函数为**==
```math
J(\beta)=1/2*\sum_{i=1}^m(h_{\beta}(x_i)-y_i)^2  

min J(\beta)
```
损失函数越小，模型的效果越好  
使用梯度下降法或最小二乘法来计算出参数
![image](/img/lr/min.jpg)

损失函数推导过程
![image](/img/lr/error.jpg)
- 梯度下降法
基本思想按照梯度下降的方向，更新参数
1. 随机初始化参数 
```math
\beta \ b
```
2. 感知器训练法则
```math
w_i \leftarrow w_i +\Delta w_i  

\Delta w_i=\eta(t-o)x_i
```
*感知器法则可以成功找到一个权向量，但如果样例不是线性可分时，它将不可收敛*；  
引入detla法则，是反向传播的基础，对误差函数E按梯度下降方向搜索，反复修改权向量，直到得到全局的最小误差点。  
```math
w_i \leftarrow w_i +\Delta w_i  

\Delta w_i=-\eta\bigtriangledown E(w_i)
```
所以，模型中参数更新方式为：
```math
\beta_j:=\beta_j-\alpha \bigtriangledown J(\beta_j)  

:=\beta_j-\alpha (-(y_i-h(x_i)))x_i  

:= \beta_j+\alpha (y_i-h(x_i)x_i

```
- 最小二乘法  
推导过程：
![image](http://images.cnitblog.com/blog2015/633472/201503/262057197703308.jpg)   
当
a.矩阵满秩可求解时（求导等于0）
  利用最小二乘法  
b. 不满秩时；使用梯度下降

实验实现
-------


小结
----
线性回归是非常简单的模型，可用来拟合现实中的一些变量关系，主要适应于连续变量；  
同时其y值存在无限延伸情况，不适用于分类问题；  
如需分类，则需要用==逻辑回归(Logistic Regression)==

参考资料
----
[线性回归详解](https://blog.csdn.net/qq_36330643/article/details/77649896)  
[从零开始学习机器学习](http://blog.51cto.com/12133258/2051527)

