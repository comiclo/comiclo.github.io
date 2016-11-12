---
layout: post
title: 使用 ipywidgets 在 jupyter notebook 建立互動式科學繪圖
tags: Python
---

匯入需要的function(interact)和class(fixed, FloatSlider)

```python
from ipywidgets import interact, fixed, FloatSlider
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

先試個簡單的函數$$f_{a,b}(x) = a \sin(x) + b \cos(x)$$

如果我們想要透過改變$$a$$和$$b$$來看$$f_{a,b}(x)$$的圖形

可以定義以下的function

```python
def f(a, b):
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])

    x = np.linspace(-10, 10, 500)
    y = np.sin(a * x) + np.cos(b * x)
    plt.plot(x, y)
```

然後使用interact, 告訴他變數變動時所要執行的function, 變數名稱和其可變動的範圍. 其中(-10, 10)表示 $[-10, 10]$ 內所有的整數.

若要需要浮點數, 可以使用FloatSlider.

```python
interact(f, a=(-10, 10), b=FloatSlider(min=1e-2, max=10, step=1e-2))
```

![](https://i.imgur.com/OeknfLH.png)

若只想變動一個變數, 固定另一個變數, 可使用fixed.

```python
interact(f, a=(-10, 10), b=fixed(1))
```

![](https://i.imgur.com/sjeCk8d.png)

再來試試看拿iris dataset餵給gradient boosting和svm, 並把幾個參數拿出來拉一拉.

```python
from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import xgboost as xgb
```

跟scikit-learn的文件一樣, 只取前兩個維度.

若使用PCA, 互動圖會比較沒感覺...

```python
iris = load_iris()
data, target = iris.data, iris.target
# data = PCA(n_components=2).fit_transform(data)
data = data[:,:2]
plt.scatter(*data.transpose(), c=target, cmap=plt.cm.coolwarm)
```
![](https://i.imgur.com/d1SJr16.png)


要畫輪廓圖, 代表預測的結果. 先準備好需要的變數

```python
h = 0.02

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
```

先試gradient boosting, 使用xgboost的XGBClassifier

```python
def gb_clf(**args):
    clf = xgb.XGBClassifier(**args)
    clf.fit(data, target)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(*data.transpose(), c=target, cmap=plt.cm.coolwarm)
    plt.text(x_min + 0.25, y_max - 0.25, 'score: ' + str(clf.score(data,target)))

```

FloatSlider似乎只能到1e-2, 不能再處理更小的了.

```python
interact(gb_clf,
         max_depth=(1, 20),
         learning_rate=FloatSlider(min=1e-2, max=1, step=1e-2),
         n_estimators=(10, 1000))
```


![](http://i.imgur.com/UW4EZ72.png)

再來是SVM

```python
def svm_clf(**args):
    clf = SVC(**args)
    clf.fit(data, target)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(*data.transpose(), c=target, cmap=plt.cm.coolwarm)
    plt.text(x_min + 0.25, y_max - 0.25, 'score: ' + str(clf.score(data,target)))

```

在調大RBF kernel的參數gamma時, 可以看到很精彩的overfitting.

```python
interact(svm_clf,
         kernel=fixed('rbf'),
         C=FloatSlider(min=1e-2, max=2, step=1e-2),
         gamma=FloatSlider(min=1e-2, max=100, step=1e-2))
```

![](http://i.imgur.com/BjxRNgP.png)

當然要做model selection時, 我們有好用的GridSearchCV...
不過若要從圖形中觀察某些參數的變化所帶來的影響, 這是一個不錯的工具, 寫起來簡單又好玩.

# Reference
- http://ipywidgets.readthedocs.io/
- http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
