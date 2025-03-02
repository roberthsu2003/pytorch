## pytorch 是Machine Learning Framework
- Deep Learning primitive
- NN Layer types
- activation & loss functions
[the role of the activation function](https://www.mropengate.com/2017/02/deep-learning-role-of-activation.html)

[Activation Function 是什麼?](./https://medium.com/%E6%B7%B1%E6%80%9D%E5%BF%83%E6%80%9D/ml08-activation-function-%E6%98%AF%E4%BB%80%E9%BA%BC-15ec78fa1ce4)
- optimizer

## RESEARCH PROTOTYPING

- Models are Python code, Autgrad and `eagermode` for dynamic model architectures


### 簡潔解釋：
- **模型是 Python 代碼**：模型是用 Python 語言寫出來的程式，定義了模型的結構和運作方式。
- **Autograd 和 `eagermode` 支持動態模型架構**：
  - **Autograd**：自動計算梯度的工具，幫助模型在訓練時調整參數。
  - **`eagermode`**：一種立即執行代碼的模式，讓模型結構能在運行時靈活變化。
  - **動態模型架構**：模型的設計可以在執行過程中動態調整，而不是事先固定。

### 總結：
這句話的意思是：模型是用 Python 寫的，而 Autograd 和 eager mode 是幫助實現動態模型的工具，讓模型結構能隨時調整並自動計算訓練所需的梯度。


## TENSORS

### 1. **Tensor 的基本概念**
在 PyTorch 中，**tensor** 是最核心的數據結構。它類似於 NumPy 的多維數組（`ndarray`），但功能更強大。簡單來說，tensor 是一個可以存儲多維數據的容器，適用於不同維度的數據，例如：

- **0 維**：純量（scalar），如單個數字 `3.14`
- **1 維**：向量（vector），如 `[1, 2, 3]`
- **2 維**：矩陣（matrix），如 `[[1, 2], [3, 4]]`
- **3 維或更高維**：高維張量，例如表示圖像數據的 `(channels, height, width)`

---

### 2. **與 NumPy 的異同**
#### **共同點**
- Tensor 和 NumPy 的數組都可以進行基本的數學運算（如加減乘除）、索引和切片操作。

#### **不同點**
- **GPU 支持**：tensor 可以在 GPU 上運行，這使得它在深度學習中能夠加速計算，而 NumPy 數組只能在 CPU 上運行。
- **自動微分**：tensor 與 PyTorch 的 Autograd 系統緊密結合，可以自動計算梯度，這對於訓練神經網絡非常重要，而 NumPy 不具備這個功能。

---

### 3. **Tensor 的重要屬性**
Tensor 有幾個關鍵屬性，幫助我們理解和操作它：

- **shape**：表示 tensor 的形狀（即每個維度的大小）。例如，一個 3 行 4 列的 tensor 其 `shape` 是 `(3, 4)`。
- **dtype**：表示 tensor 中數據的類型，例如 `float32`（浮點數）、`int64`（整數）等。
- **device**：表示 tensor 所在的設備，可以是 CPU（默認）或 GPU（例如 `cuda:0`）。

你可以用以下代碼查看這些屬性：
```python
import torch
tensor = torch.tensor([[1, 2], [3, 4]])
print(tensor.shape)  # 輸出：torch.Size([2, 2])
print(tensor.dtype)  # 輸出：torch.int64
print(tensor.device) # 輸出：cpu
```

---

### 4. **創建 Tensor 的方法**
PyTorch 提供了多種創建 tensor 的方法，以下是常見的幾種：

#### **直接從數據創建**
```python
import torch
data = [1, 2, 3]
tensor = torch.tensor(data)  # 創建一個 1 維 tensor
print(tensor)  # 輸出：tensor([1, 2, 3])
```

#### **從 NumPy 數組創建**

```python
import numpy as np
array = np.array([1, 2, 3])
tensor = torch.from_numpy(array)
print(tensor)  # 輸出：tensor([1, 2, 3])
```

#### **創建特殊 Tensor**
- 全 0 tensor：`torch.zeros((2, 3))` 創建一個 2x3 的全零 tensor。
- 全 1 tensor：`torch.ones((2, 3))` 創建一個 2x3 的全一 tensor。
- 隨機 tensor：`torch.rand((2, 3))` 創建一個 2x3 的隨機數 tensor（值在 0 到 1 之間）。

---

### 5. **Tensor 的運算**
Tensor 支持多種運算，這些運算是深度學習的基礎：

#### **基本運算**
加減乘除等操作可以直接應用於 tensor：

```python
tensor1 = torch.tensor([1, 2])
tensor2 = torch.tensor([3, 4])
result = tensor1 + tensor2  # 輸出：tensor([4, 6])
```

#### **矩陣運算**
例如矩陣乘法：

```python
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(tensor1, tensor2)
print(result)  # 輸出：tensor([[19, 22], [43, 50]])
```

#### **聚合運算**
例如求和或平均：

```python
tensor = torch.tensor([1, 2, 3])
print(tensor.sum())   # 輸出：tensor(6)
print(tensor.mean())  # 輸出：tensor(2.)
```

---

### 6. **Tensor 與 Autograd（自動微分）**
Tensor 在深度學習中最重要的特性之一是支持自動微分，這得益於 PyTorch 的 Autograd 系統。

- **requires_grad**：當設置為 `True` 時，tensor 會記錄它的運算歷史，以便計算梯度。
- **grad**：存儲該 tensor 的梯度值。

範例：
```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()  # 計算梯度
print(x.grad)  # 輸出：tensor([2.0])
```
這裡，`y = 2 * x`，對 `y` 求導得到 `dy/dx = 2`，因此 `x.grad` 為 2.0。

---

### 7. **Tensor 在深度學習中的應用**
Tensor 是 PyTorch 構建和訓練深度學習模型的基礎，應用場景包括：

- **數據輸入**：圖像、文本等數據會被轉換為 tensor 格式，作為模型的輸入。
- **模型參數**：神經網絡的權重和偏置都以 tensor 的形式存儲。
- **訓練過程**：通過 tensor 的運算實現前向傳播，通過自動微分實現反向傳播，更新模型參數。

---

### **總結**
**Tensor** 是 PyTorch 中處理多維數據的基礎工具，類似於 NumPy 的多維數組，但它具備以下優勢：
1. 支持 GPU 加速，提升計算效率。
2. 與 Autograd 結合，支持自動微分，方便訓練神經網絡。

簡單來說，tensor 不僅是一個數據容器，更是深度學習模型的核心組件。無論是數據處理、模型參數還是訓練過程，tensor 都扮演著不可或缺的角色。希望這個解釋能讓你對 PyTorch 的 tensor 有更深入的理解！如果還有疑問，歡迎繼續提問。

## Autograd
當然可以！以下是對 PyTorch 中 **Autograd** 的基礎說明和用途的詳細解釋，讓你能夠全面理解它的核心概念和在深度學習中的重要作用。

---

## Autograd 的基礎說明

### 什麼是 Autograd？
Autograd 是 PyTorch 提供的一個核心功能，全稱為 **Automatic Differentiation**（自動微分）。它的主要任務是**自動計算梯度**（gradient），也就是函數對其變量的偏導數。在深度學習中，梯度是用來指導模型參數如何調整以減少損失（loss）的關鍵信息。有了 Autograd，你無需手動推導複雜的數學公式，它會自動幫你完成梯度計算。

簡單來說，Autograd 就像一個「記錄員」和「計算器」：它會記錄你對數據（tensor）執行的所有運算，並在需要時根據這些記錄計算出梯度。

### 工作原理：計算圖（Computational Graph）
Autograd 的核心是基於**計算圖**的運作方式。計算圖是一種用來表示數學運算過程的結構，具體來說：

- **節點（Node）**：代表 tensor（張量），也就是數據本身。
- **邊（Edge）**：代表運算，例如加法、乘法、冪運算等。

當你對 tensor 進行運算時，Autograd 會在後台動態構建一個計算圖，記錄這些運算的順序和依賴關係。這個計算圖是**動態的**，意味著它是在代碼運行時即時生成的，而不是事先定義好的，這也是 PyTorch 相較於其他框架（如 TensorFlow 的靜態圖）的靈活之處。

#### 前向傳播（Forward Pass）
在前向傳播中，你將輸入數據通過一系列運算（例如神經網絡層）轉換成輸出。Autograd 會默默記錄下每一步運算，形成計算圖，但此時並不計算梯度。

#### 反向傳播（Backward Pass）
當你需要計算梯度時（通常是在計算損失函數後），Autograd 會從計算圖的終點（例如損失值）開始，沿著圖反向計算每一個 tensor 的梯度。這依賴於**鏈式法則**（chain rule），Autograd 會自動處理所有中間步驟的導數計算。

---

### 關鍵概念
要理解 Autograd 的基礎，以下幾個概念非常重要：

1. **`requires_grad`**
   - 這是 tensor 的一個屬性，默認值為 `False`。
   - 當設置為 `True` 時，Autograd 會追蹤該 tensor 上的所有運算，並將其納入計算圖。
   - 例如，模型的參數（如權重和偏置）通常需要設置 `requires_grad=True`，因為我們需要計算它們的梯度來更新參數。

2. **`grad_fn`**
   - 當一個 tensor 是通過運算生成的，它會有一個 `grad_fn` 屬性，記錄產生這個 tensor 的運算。
   - 例如，如果 `y = x * 2`，那麼 `y.grad_fn` 會指向乘法運算。

3. **`backward()`**
   - 這是 tensor 的一個方法，用來觸發反向傳播，計算梯度。
   - 當你對某個 tensor（通常是損失值）調用 `backward()` 時，Autograd 會根據計算圖計算所有設置了 `requires_grad=True` 的 tensor 的梯度。

4. **`grad`**
   - 這是 tensor 的屬性，用來存儲該 tensor 的梯度值。
   - 在調用 `backward()` 後，相關 tensor 的 `grad` 屬性會被填充上計算出的梯度。

---

### 簡單範例
我們通過一個具體例子來說明 Autograd 的運作過程：

```python
import torch

# 創建一個 tensor，設置 requires_grad=True
x = torch.tensor([2.0], requires_grad=True)

# 定義運算
y = x * 3  # y = 3x
z = y ** 2  # z = (3x)^2 = 9x^2

# 觸發反向傳播
z.backward()

# 查看梯度
print(x.grad)  # 輸出：tensor([36.])
```

#### 解釋：
1. **前向傳播**：
   - 初始值：`x = 2.0`。
   - 計算 `y = 3 * x = 6.0`。
   - 計算 `z = y^2 = 36.0`。
   - Autograd 記錄下這些運算，形成計算圖。

2. **反向傳播**：
   - 調用 `z.backward()`，計算 `z` 對 `x` 的梯度。
   - 使用鏈式法則：
     - `dz/dy = 2y = 2 * 6 = 12`。
     - `dy/dx = 3`。
     - 因此，`dz/dx = dz/dy * dy/dx = 12 * 3 = 36`。
   - 結果：`x.grad = 36.0`。

這個例子展示了 Autograd 如何自動計算梯度，無需我們手動推導。

---

## Autograd 的用途

### 在深度學習中的核心作用
在深度學習中，Autograd 的主要用途是**自動計算損失函數對模型參數的梯度**，從而實現參數的優化和模型的訓練。以下是它在訓練過程中的具體應用：

#### 訓練過程
1. **前向傳播**：
   - 將輸入數據送入模型，通過層層運算得到預測輸出。
   - Autograd 記錄所有運算步驟。

2. **計算損失**：
   - 使用預測輸出和真實標籤計算損失函數（例如均方誤差或交叉熵）。

3. **反向傳播**：
   - 對損失函數調用 `backward()`，Autograd 自動計算每個參數的梯度。
   - 這些梯度告訴我們每個參數如何影響損失。

4. **更新參數**：
   - 使用優化器（例如 SGD 或 Adam）根據梯度更新模型參數，逐步減小損失。

#### 實際例子
假設你有一個簡單的神經網絡，輸入是 `x`，參數是 `w` 和 `b`，輸出是 `y = w * x + b`，損失是 `loss = (y - target)^2`。在訓練中：
- Autograd 會計算 `loss` 對 `w` 和 `b` 的梯度。
- 優化器根據這些梯度調整 `w` 和 `b`，使損失逐漸減小。

---

### 為什麼 Autograd 重要？
Autograd 在深度學習中有以下幾個關鍵優勢：

1. **自動化**：
   - 深度學習模型通常包含數十萬甚至數百萬個參數，手動計算梯度幾乎不可能。Autograd 自動完成這一過程，大大簡化開發。

2. **靈活性**：
   - 動態計算圖允許你在運行時改變模型結構（例如根據輸入數據動態調整層數），這在研究和開發中非常有用。

3. **高效性**：
   - Autograd 內部優化了梯度計算過程，結合 PyTorch 的高效 tensor 運算，保證了訓練速度。

---

### 總結
**Autograd** 是 PyTorch 中實現自動微分的核心工具。它通過動態構建計算圖，追蹤 tensor 上的運算，並在反向傳播時自動計算梯度。它的基礎原理依賴於計算圖和鏈式法則，主要用途是幫助深度學習模型在訓練過程中計算參數梯度，從而實現參數更新和損失優化。無論是簡單的線性回歸還是複雜的神經網絡，Autograd 都讓梯度計算變得簡單、高效，是 PyTorch 易用性和靈活性的重要支柱。
