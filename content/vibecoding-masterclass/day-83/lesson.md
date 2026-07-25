哈囉，各位未來的大大們！歡迎來到我們的 MLOps 學習旅程「第 83 天」！

恭喜你們走到這裡！我們已經一起走過模型訓練、資料處理等許多精彩的環節。但大家有沒有想過，當你的模型超級厲害，表現得無懈可擊時，要怎麼才能穩定、快速地把這個「寶藏」交到使用者手上，讓它為大家服務呢？

這時候，就是我們今天的超級英雄登場啦——**MLOps 持續整合 (CI) 與持續部署 (CD)**！

---

## 【第 83 天：實戰：MLOps 持續整合與持續部署 (CI/CD)】

### 🚀 什麼是 MLOps CI/CD？

想像一下你正在生產一輛超酷的自動駕駛汽車模型。

*   **持續整合 (Continuous Integration, CI)**：
    *   就像是你和你的團隊，每次對汽車設計圖（程式碼）做了任何小改動，都會立刻把它們整合到一個主要的設計圖版本中。
    *   然後，自動化的機器人會立即檢查這些改動有沒有造成任何問題，例如：新的自動駕駛演算法會不會導致車子突然撞牆？資料預處理的流程是不是還能正常運作？測試資料跑出來的結果有沒有問題？
    *   **在 MLOps 中，CI 確保了你的程式碼、資料處理管道、模型訓練邏輯，甚至連模型本身，在每次更動後都能保持高品質，沒有引入新的錯誤。**

*   **持續部署 (Continuous Deployment, CD)**：
    *   如果 CI 機器人確認了新的設計圖是完美無瑕的，那麼 CD 就像是一個全自動化的生產線，會立刻把這輛最新的自動駕駛汽車模型直接送到展示間（生產環境），讓使用者可以直接體驗！
    *   **在 MLOps 中，CD 確保了經過驗證的模型，能夠自動化地被打包、部署，並提供給真實世界的使用者。**

簡而言之，**CI/CD 就是要讓你的 ML 專案開發流程「又快又穩」！**

### 🤔 為什麼 MLOps 需要 CI/CD？

1.  **品質保證，信心滿滿**：每次提交程式碼，都會自動執行測試，大大減少錯誤，讓你對模型的品質更有信心。
2.  **快速迭代，搶佔先機**：不再需要手動部署的漫長等待，新模型或新功能可以更快地推向市場。
3.  **減少手動錯誤**：自動化流程取代了容易出錯的手動操作。
4.  **更好的協作**：團隊成員可以更頻繁地整合他們的程式碼，減少「整合地獄」。
5.  **可重現性**：每次部署都有清晰的紀錄，方便追溯和回滾。

### 🛠️ 實戰時間！GitHub Actions 登場！

為了讓大家能實際感受 CI/CD 的魔力，我們將使用 **GitHub Actions**。它是一個非常流行且易於上手的 CI/CD 工具，直接整合在 GitHub 平台裡。

我們來建構一個簡單的範例：當你修改並推送到 `main` 分支時，會觸發 CI 流程來「測試」你的模型訓練程式碼，如果測試成功，就會觸發 CD 流程來「部署」你的模型。

#### 1. 建立專案結構

在你的 GitHub 專案根目錄下，建立以下檔案和資料夾：

```
.
├── .github
│   └── workflows
│       ├── ci.yml      # CI 工作流定義
│       └── cd.yml      # CD 工作流定義
├── requirements.txt    # 專案依賴套件
└── train.py            # 模型訓練腳本
```

#### 2. `requirements.txt` (專案依賴)

我們需要一些基本的機器學習套件：

```txt
scikit-learn
pandas
numpy
```

#### 3. `train.py` (模型訓練腳本)

這是一個非常簡單的模擬訓練腳本，重點在於它能被執行：

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
import os

def train_model():
    print("🚀 啟動模型訓練模擬...")
    
    # 模擬一些資料
    data = {'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 0, 1, 1, 1]}
    df = pd.DataFrame(data)

    X = df[['feature1', 'feature2']]
    y = df['target']

    # 簡單的邏輯迴歸模型
    model = LogisticRegression()
    model.fit(X, y)

    print("✅ 模型訓練完成！")
    
    # 在真實情境中，你會在這裡保存模型，例如：
    # import joblib
    # joblib.dump(model, 'model.joblib')
    # print("模型已保存為 model.joblib")
    return model

if __name__ == "__main__":
    # 我們可以利用 --test-run 參數來區分 CI/CD 階段
    # 在 CI 中，我們可能只運行測試而不實際保存模型
    if "--test-run" in sys.argv:
        print("💡 正在以測試模式運行 train.py (不實際保存模型)。")
        train_model()
    else:
        print("✨ 正在以部署模式運行 train.py (模擬實際訓練與保存模型)。")
        train_model()
        # 這裡可以模擬保存一個 artifact
        with open("trained_model_version.txt", "w") as f:
            f.write("model_v1.0.0")
        print("模型已模擬保存，準備部署！")

```

#### 4. `.github/workflows/ci.yml` (CI 工作流)

這個檔案定義了當程式碼被推送到 `main` 分支時，要執行的自動化測試步驟：

```yaml
name: ML CI Workflow # 工作流的名稱

on:
  push:
    branches:
      - main # 當有程式碼推送到 main 分支時觸發

jobs:
  build-and-test:
    runs-on: ubuntu-latest # 在 Ubuntu 系統上執行這個 Job

    steps:
    - name: ⬇️ 檢查程式碼
      uses: actions/checkout@v3 # 抓取 GitHub 上的專案程式碼

    - name: 🐍 設定 Python 環境
      uses: actions/setup-python@v4
      with:
        python-version: '3.9' # 指定 Python 版本

    - name: 📦 安裝依賴套件
      run: pip install -r requirements.txt # 安裝 requirements.txt 中的所有套件

    - name: 🧪 執行模型訓練腳本測試
      run: python train.py --test-run # 執行訓練腳本，檢查是否有語法或執行錯誤

    - name: ✅ 輸出 CI 測試結果
      run: echo "🎉 CI 測試成功！程式碼品質良好，準備好進行後續部署啦！"

```

#### 5. `.github/workflows/cd.yml` (CD 工作流)

這個檔案定義了當 **CI 工作流成功完成** 後，要執行的自動化部署步驟：

```yaml
name: ML CD Workflow # 工作流的名稱

on:
  workflow_run: # 當另一個 workflow 成功完成時觸發
    workflows: ["ML CI Workflow"] # 監聽名為 "ML CI Workflow" 的 workflow
    types:
      - completed # 當它完成時
    branches:
      - main # 並且是在 main 分支上完成的

jobs:
  deploy-model:
    runs-on: ubuntu-latest
    # 只有當 CI 工作流的結果是 'success' 時才執行這個 Job
    if: ${{ github.event.workflow_run.conclusion == 'success' }} 

    steps:
    - name: ⬇️ 檢查程式碼
      uses: actions/checkout@v3 

    - name: 🐍 設定 Python 環境
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: 📦 安裝依賴套件
      run: pip install -r requirements.txt

    - name: 🚀 部署模型 (模擬操作)
      run: |
        echo "🎉 CI 流程已成功完成，現在開始部署模型！"
        echo "這一步將模擬在生產環境中實際訓練和部署模型。"
        python train.py # 在此處執行實際的模型訓練和保存
        echo "模型已成功訓練並保存為 artifact (例如：上傳到 S3/GCS 或是更新模型服務端點)！"
        echo "你可以想像這一步是將新的模型版本推送到：一個模型註冊中心、一個 API 服務，或是容器化部署到 Kubernetes 等。"
        echo "✨ 模型部署完成！恭喜！你的新模型已經為用戶服務了！"
```

### 🧠 它是如何運作的？

1.  當你修改 `train.py` 或 `requirements.txt`，並推送到 GitHub 專案的 `main` 分支時。
2.  **`ci.yml`** 工作流會被自動觸發。它會設定 Python 環境，安裝依賴，並執行 `python train.py --test-run` 來檢查你的訓練程式碼是否能正常運行，沒有語法錯誤或崩潰。
3.  如果 `ci.yml` 成功執行完畢（所有步驟都綠色打勾），那麼 **`cd.yml`** 工作流就會被自動觸發。
4.  `cd.yml` 會再次設定環境，然後執行 `python train.py`（這次是模擬實際的訓練和「部署」操作）。在真實世界中，這裡可能會包含將訓練好的模型上傳到模型註冊中心、打包成 Docker 映像檔並部署到 Kubernetes 叢集，或是更新一個 API 服務端點等。

---

### 🎉 恭喜你！

現在你已經對 MLOps 的 CI/CD 有了初步的實戰經驗！這是一個非常強大的概念，它能讓你的 ML 專案從開發到部署的過程變得更順暢、更可靠。

一開始可能會覺得 YAML 設定檔有點複雜，但這就是自動化的魔法。多加練習，你會發現它們非常直觀。繼續探索，你會發現 MLOps 的樂趣無窮！

下次見！Keep coding, keep learning！🚀