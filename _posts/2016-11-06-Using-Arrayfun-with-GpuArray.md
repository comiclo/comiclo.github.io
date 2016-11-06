---
layout: post
title: Using Arrayfun with GpuArray
---

當我們需要把矩陣中的每一個元素都送進一個實數到實數的函數時，應該避免使用for loop。我們可以使用簡單的技巧來達到我們的目的，例如：
```
(x .^ 2) .* double(x > 0)
```
運算時MATLAB會做平行化的動作，但這樣的方法不是很理想。

MATLAB提供了arrayfun，我們可以定義$f:\mathbb{R} \rightarrow \mathbb{R} $的函數，並且讓MATLAB改成$F:\mathbb{R}^{n \times m} \rightarrow \mathbb{R}^{n \times m}$
的函數，其中$F(X)_{i,j} = f(X_{i,j})$，這樣可以使程式碼更容易閱讀，速度似乎也比較快。

另外numpy也有類似的方法
https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html

# 使用範例

定義函數

```
function y = fun(x)
if x > 0
    y = x^2;
else
    y = 0;
end
```

主程式

```
x = randn(1000, 1000, 'gpuArray');

f = @() (x .^ 2) .* double(x > 0);
g = @() arrayfun(@fun, x);

[timeit(f), timeit(g)]
```

確認兩個函數產生的結果是一樣的

```
norm((x .^ 2) .* double(x > 0)  - arrayfun(@fun, x), 'fro')
```

# Reference
https://www.mathworks.com/help/distcomp/arrayfun.html
https://www.mathworks.com/help/distcomp/run-element-wise-matlab-code-on-a-gpu.html


