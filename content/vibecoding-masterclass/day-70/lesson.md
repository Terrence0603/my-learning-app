恭喜你！來到程式學習的第 70 天，這是一個非常特別的里程碑！今天我們不只會寫程式，還要將你的機器學習技能提升到一個新的境界，進入令人興奮的 **MLOps (Machine Learning Operations)** 世界！

別擔心，聽起來很專業對吧？但其實 MLOps 的核心精神，就是讓你的機器學習專案，從數據到部署，都能更順暢、更自動化、更可靠。今天，我們將親手建構一個簡單的「端到端 (End-to-End)」MLOps 自動化管線。

## 主題：【第 70 天：實戰：建構端到端 MLOps 自動化管線】

想像一下，你辛苦訓練出一個很棒的模型，但每次有新數據、需要重新訓練時，你都要手動跑一次程式、手動保存模型、再手動部署嗎？太麻煩啦！MLOps 就是要解決這個痛點，讓這個過程「自動化」！

今天，我們的目標是：
1.  **訓練模型**：使用 Python 和 `scikit-learn` 訓練一個簡單的模型。
2.  **自動化管線**：利用 **GitHub Actions** 這個強大的工具，當你提交程式碼時，自動執行訓練、保存模型，甚至模擬部署的過程。

準備好了嗎？讓我們捲起袖子，動手實作吧！

### 專案結構 (Project Structure)

首先，我們需要一個整潔的專案結構。請在你的電腦上建立這些檔案和資料夾：

```
your-mlops-project/
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml  # 我們的 GitHub Actions 配置檔
├── src/
│   ├── train.py                # 負責模型訓練的程式碼
│   └── predict.py              # 負責模型預測/部署模擬的程式碼
├── requirements.txt            # 專案所需的 Python 函式庫
└── README.md                   # 專案說明 (可選)
```

### 步驟一：準備核心 ML 程式碼

我們將創建一個簡單的分類模型，並模擬它的訓練和預測過程。

**1. `requirements.txt`**

這個檔案列出了我們專案需要的所有 Python 函式庫。

```txt
scikit-learn
pandas
joblib # 用於保存和加載模型
```

**2. `src/train.py` (模型訓練)**

這個檔案負責生成一些模擬數據，訓練一個邏輯迴歸模型，並將訓練好的模型保存起來。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification # 簡化數據生成
import joblib
import os

print("--- [階段一] 開始模型訓練 ---")

# 1. 數據準備 (這裡我們簡化為生成數據)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"訓練數據形狀: {X_train.shape}, 測試數據形狀: {X_test.shape}")

# 2. 模型訓練
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("模型訓練完成！")

# 3. 模型評估 (簡化)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在測試集上的準確度: {accuracy:.4f}")

# 4. 保存模型
model_dir = "model" # 模型將保存到 'model/' 資料夾
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "logistic_regression_model.joblib")
joblib.dump(model, model_path)
print(f"模型已保存至: {model_path}")

print("--- [階段一] 模型訓練結束 ---")
```

**3. `src/predict.py` (模型預測/部署模擬)**

這個檔案負責加載之前保存的模型，並用一些新數據來模擬預測過程。在 MLOps 管線中，這部分程式碼可能會變成一個 API 服務或批次預測服務。

```python
import joblib
import numpy as np
import os

print("--- [階段二] 開始模型預測模擬 ---")

# 1. 加載模型
model_path = os.path.join("model", "logistic_regression_model.joblib")
if not os.path.exists(model_path):
    print(f"錯誤：模型檔案 '{model_path}' 不存在。請確保訓練流程已執行並保存模型。")
    exit(1)

model = joblib.load(model_path)
print(f"模型已從 {model_path} 加載成功。")

# 2. 模擬新數據進行預測
# 這裡我們簡單生成一個新的數據點
new_data = np.random.rand(1, 10) # 1個樣本，10個特徵
print(f"模擬輸入數據: {new_data}")

prediction = model.predict(new_data)
prediction_proba = model.predict_proba(new_data)

print(f"模型預測結果: {prediction[0]}")
print(f"模型預測機率: {prediction_proba[0]}")
print("--- [階段二] 模型預測模擬結束 ---")
```

### 步驟二：建構 GitHub Actions 自動化管線

現在，ML 程式碼準備好了，是時候讓它「自動」跑起來了！我們將使用 GitHub Actions 來實現這一點。

**1. `mlops_pipeline.yml` (GitHub Actions 配置)**

這個 YAML 檔案定義了當我們推送到 GitHub 倉庫的 `main` 分支時，應該執行哪些步驟。

```yaml
name: MLOps End-to-End Pipeline # 給你的工作流一個名字

on:
  push:
    branches:
      - main # 當有代碼推送到 main 分支時觸發這個工作流

jobs:
  build-and-train: # 第一個 Job：負責建置環境和訓練模型
    runs-on: ubuntu-latest # 在最新的 Ubuntu 虛擬機上運行

    steps:
    - name: 檢查代碼 (Checkout code)
      uses: actions/checkout@v3 # 下載你的 GitHub 倉庫代碼

    - name: 設定 Python 環境
      uses: actions/setup-python@v4 # 安裝指定版本的 Python
      with:
        python-version: '3.9' # 指定 Python 版本，確保與你的環境相符

    - name: 安裝依賴 (Install dependencies)
      run: | # 執行多個 shell 命令
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: 運行模型訓練 (Run model training)
      run: python src/train.py # 執行你的訓練腳本

    - name: 上傳訓練好的模型 (Upload trained model artifact)
      uses: actions/upload-artifact@v3 # 將模型作為「artifact」上傳
      with:
        name: trained-model # 給這個 artifact 一個名字
        path: model/logistic_regression_model.joblib # 告訴 Actions 模型在哪裡

  deploy-and-test: # 第二個 Job：負責模擬部署和測試
    runs-on: ubuntu-latest
    needs: build-and-train # 這個 Job 必須在 'build-and-train' 成功後才運行

    steps:
    - name: 檢查代碼 (Checkout code) # 需要獲取 predict.py
      uses: actions/checkout@v3

    - name: 下載訓練好的模型 (Download trained model artifact)
      uses: actions/download-artifact@v3 # 下載上一個 Job 上傳的模型
      with:
        name: trained-model # 指定要下載的 artifact 名稱
        path: model # 模型會被下載到這個目錄 (與 train.py 保存的目錄結構一致)

    - name: 設定 Python 環境
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: 安裝依賴 (Install dependencies for predict)
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt # 預測腳本也可能需要這些依賴

    - name: 運行模型預測模擬 (Run model prediction simulation)
      run: python src/predict.py # 執行你的預測腳本
```

### 步驟三：推送到 GitHub 並觀察奇蹟發生！

1.  **初始化 Git 倉庫**：
    在你的專案根目錄 (your-mlops-project/) 中打開終端機，執行：
    ```bash
    git init
    git add .
    git commit -m "feat: Initial MLOps pipeline setup"
    ```
2.  **在 GitHub 上建立一個新的私人或公開倉庫**。
3.  **將你的本地倉庫連結到 GitHub**：
    ```bash
    git remote add origin https://github.com/你的用戶名/你的倉庫名.git
    git branch -M main
    git push -u origin main
    ```
4.  **觀察 GitHub Actions**：
    現在，打開你的 GitHub 倉庫頁面，點擊上方的 "Actions" 選項卡。你應該會看到一個正在運行的工作流，它的名字就是 `MLOps End-to-End Pipeline`。點擊進去，你會看到 `build-and-train` 和 `deploy-and-test` 這兩個 Job 正在依序執行。

    當它們都成功運行完畢，恭喜你！你已經成功建立了一個端到端的 MLOps 自動化管線！

### 恭喜你！你已經是 MLOps 的英雄了！

**你剛才做了什麼？**
*   你編寫了可以訓練和預測機器學習模型的程式碼。
*   你配置了一個 GitHub Actions 工作流，讓它能夠自動：
    *   檢查你的程式碼。
    *   設置一個乾淨的 Python 環境。
    *   安裝所有必要的函式庫。
    *   運行模型訓練腳本，並將訓練好的模型保存為「artifact」。
    *   自動下載這個模型 artifact。
    *   運行模型預測腳本，模擬模型的部署和使用。

這就是 MLOps 的魅力！每次你改動了訓練程式碼，或者有了新的數據（雖然這次是模擬的），你只需要 `git push`，剩下的所有步驟都會自動化執行。這極大地提高了機器學習專案的開發效率、可重複性和可靠性。

### 未來展望

這只是一個非常基礎的 MLOps 管線。在真實世界中，你可能會進一步探索：
*   **數據版本控制 (Data Versioning)**：例如使用 DVC。
*   **模型註冊 (Model Registry)**：更好地管理不同版本的模型。
*   **模型監控 (Model Monitoring)**：監控部署模型的效能。
*   **更複雜的部署**：將模型部署為真正的 API 服務 (例如使用 Flask、FastAPI) 或雲端服務 (AWS SageMaker, Google AI Platform, Azure ML)。
*   **單元測試和整合測試**：確保 ML 程式碼的品質。

第 70 天的挑戰非常精彩，你做得太棒了！繼續保持這樣探索的精神，機器學習的世界會因為有你而更加強大！加油！