在神经网络训练过程中，损失函数和优化器是两个关键的组成部分，它们分别用于度量模型的性能和更新模型的参数。

### 损失函数（交叉熵损失）：

损失函数用于度量模型输出与实际标签之间的差异，是优化算法的目标。交叉熵损失在分类问题中是一种常见的损失函数，尤其是对于多类别分类任务。

在PyTorch中，`nn.CrossEntropyLoss`经常用于多类别分类问题。交叉熵损失的计算公式如下：

\[ \text{CrossEntropyLoss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(\hat{y}_i) \right) \]

其中，\(N\) 是样本数量，\(y_i\) 是实际标签的独热编码（或类别索引），\(\hat{y}_i\) 是模型的预测输出，\(\log\) 是自然对数。交叉熵损失旨在最小化模型输出与实际标签之间的差异。

### 优化器（随机梯度下降SGD）：

优化器用于更新神经网络的参数，以最小化损失函数。随机梯度下降（SGD）是最简单和最常用的优化算法之一。其基本思想是通过计算损失函数关于参数的梯度，然后按照梯度的方向更新参数，使得损失函数逐渐减小。

在PyTorch中，`torch.optim.SGD`是实现随机梯度下降的优化器。优化器的初始化通常需要指定模型的参数和学习率。下面是一个简单的例子：

```python
import torch
import torch.optim as optim

# 创建模型实例（假设模型名为 model）
model = SimpleCNN()

# 定义损失函数（交叉熵损失）
criterion = nn.CrossEntropyLoss()

# 定义优化器（随机梯度下降SGD）
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

在训练过程中，你会在每个训练迭代中执行以下步骤：

```python
# 清零梯度
optimizer.zero_grad()

# 前向传播
output = model(input_data)

# 计算损失
loss = criterion(output, target)

# 反向传播
loss.backward()

# 参数更新
optimizer.step()
```

这样，通过反复迭代，模型的参数将会根据梯度下降的策略逐渐调整，使得损失函数最小化。