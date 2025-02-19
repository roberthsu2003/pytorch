# PyTorch
- 基礎pytorch,主要無縫進入transformer學習
- [pytorch官方教學網站](https://pytorch.org/tutorials/)

## PyTorch基礎
### [pyTorch Tensors](./pyTorch_Tensors)
- PyTorch 中的核心數據結構，類似於 NumPy 陣列，但支援 GPU 運算。
-  需要理解 .cuda() 如何將張量搬到 GPU。
 
### [Autograd(自動微分)](./Autograd)
- 反向傳播（backpropagation）概念。
- backward() 計算梯度，.zero_grad() 重置梯度。

### [Optimizers(優化器)](./Optimizers)
- 作用：更新模型權重（如 Adam）。
- optimizer.step() 用來執行梯度下降。

### [Loss function(損失函數)](./Loss_function)
- 定義單個訓練樣本與真實值之間的誤差

## PyTorch深度學習基礎

### [建立一個簡單的神經網路](./簡單的神經網路)

### [訓練與測試模型](./訓練和測試模型)


## PyTorch與NLP

### [使用DataLoader載入NLP數據](./DataLoader)

### [PyTorch處理NLP文本](./PyTorch處理NLP文本)


