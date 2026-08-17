恭喜你來到【第 106 天】！🎉 走到這一步，你對機器學習的程式碼、模型訓練想必已經駕輕就熟了。但你知道嗎？光會寫程式、訓練模型還不夠喔！當你的模型要真正應用到實際環境時，還需要一套「魔法」，讓整個流程更順暢、更可靠、更自動化。這套魔法就叫做 **MLOps**！

今天，我們要一起揭開 MLOps 的神秘面紗，專注於它的核心環節之一：**自動化 CI/CD 流水線的建構**。聽起來有點硬核，對不對？別擔心，我會用最輕鬆、最白話的方式，帶你一步步理解它，甚至親手搭建一個簡單的自動化流程！

---

## 【第 106 天：實戰：MLOps 自動化 CI/CD 流水線建構】

### 🌟 什麼是 MLOps？為什麼我們需要 CI/CD？

想像一下，你辛辛苦苦訓練出一個超棒的模型，可以準確預測房價、辨識圖片，或者推薦商品。現在，你需要把這個模型部署到線上，讓大家都能用。這個過程通常會遇到這些挑戰：

1.  **程式碼更新：** 團隊成員修改了模型程式碼、前處理邏輯，每次都要手動測試、重新訓練、重新部署嗎？
2.  **數據變更：** 新數據進來了，模型需要重新訓練，怎麼確保每次訓練都用正確的數據和參數？
3.  **環境一致性：** 開發環境、測試環境、部署環境的套件版本、Python 版本不同步，導致模型在這裡跑得好，到那裡就出問題？
4.  **模型版本管理：** 部署了新模型，舊模型怎麼辦？如果新模型效果不好，能不能快速回溯到舊版本？

這就是 MLOps 登場的時機！

**MLOps (Machine Learning Operations)** 簡單來說，就是把傳統軟體開發中的 **DevOps 理念** 應用到 **機器學習專案** 中。它旨在透過自動化、標準化和監控，來提升機器學習模型的開發、部署和維護效率。

而 **CI/CD (Continuous Integration / Continuous Delivery)** 則是 MLOps 的核心支柱。

*   **CI (持續整合):** 每次你或你的團隊成員提交新的程式碼時，系統會自動跑一連串的測試（例如：程式碼風格檢查、單元測試、數據驗證、模型基本功能測試），確保新的變更沒有破壞現有的功能。
*   **CD (持續交付/部署):** 如果 CI 的所有測試都通過了，系統可以自動將模型打包、部署到測試環境，甚至直接部署到生產環境。

有了 CI/CD，你就可以告別手動、重複、容易出錯的部署流程，讓你的 ML 專案像坐上火箭一樣，又快又穩！🚀

### 🛠️ 我們的工具：GitHub Actions

在眾多 CI/CD 工具中，我們今天選擇一個對初學者非常友善，且與 GitHub 緊密整合的工具：**GitHub Actions**。

GitHub Actions 讓你可以在你的 GitHub 程式碼倉庫中，直接定義自動化工作流程 (workflow)。它使用 YAML 語法來描述這些流程，非常直觀。

### 🏗️ 實戰：建構一個簡單的 MLOps CI/CD 流水線

我們的目標是：當我們將模型程式碼推送到 GitHub 時，GitHub Actions 會自動幫我們：
1.  檢查程式碼。
2.  安裝必要的依賴套件。
3.  運行一個簡單的測試（證明我們的程式碼沒有大問題）。
4.  **訓練** 並 **儲存** 一個機器學習模型。
5.  將訓練好的模型作為「產物 (artifact)」保存起來。

#### 步驟一：準備你的專案

首先，在你的專案根目錄下，創建以下檔案：

1.  **`model.py` (模型訓練腳本):**
    這是一個非常簡單的 Scikit-learn 模型訓練腳本。

    ```python
    # model.py
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import joblib
    import os

    print("--- 開始訓練模型 ---")

    # 模擬生成一些數據
    df = pd.DataFrame({
        'feature1': [i for i in range(100)],
        'feature2': [i * 2 for i in range(100)],
        'target': [0 if i < 50 else 1 for i in range(100)]
    })

    X = df[['feature1', 'feature2']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 訓練一個簡單的邏輯迴歸模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"模型訓練完成，準確率: {accuracy:.2f}")

    # 儲存模型為 model.pkl
    model_path = 'model.pkl'
    joblib.dump(model, model_path)
    print(f"模型已儲存為 {model_path}")

    # 為了演示，我們也創建一個假的預測輸出檔案
    with open('predictions.txt', 'w') as f:
        f.write("這是模型預測的輸出結果 (範例)\n")
        f.write(f"範例準確率: {accuracy:.2f}\n")
    print("預測輸出檔案 predictions.txt 已建立。")

    print("--- 模型訓練流程結束 ---")
    ```

2.  **`requirements.txt` (依賴套件列表):**
    列出你的模型訓練腳本所需的套件。

    ```
    scikit-learn
    pandas
    joblib
    pytest # 我們會用來跑一個簡單的測試
    ```

3.  **`test_model.py` (簡單的測試腳本):**
    這個腳本只包含一個最基本的測試，確保 `pytest` 環境能正常運行。

    ```python
    # test_model.py
    def test_dummy_success():
        """
        一個簡單的測試，確保測試環境正常。
        在真實專案中，這裡會有針對模型邏輯、數據處理的單元測試。
        """
        assert True
        print("運行了一個假測試，成功！")

    # 你可以在這裡添加更多實際的測試
    # 例如：
    # import joblib
    # def test_model_loading():
    #     model = joblib.load('model.pkl')
    #     assert model is not None
    ```

#### 步驟二：建立 GitHub Actions Workflow

在你的專案根目錄下，創建一個資料夾 `./.github/workflows/`，然後在這個資料夾裡面創建一個 YAML 檔案，例如 `ml_pipeline.yml`。

```yaml
# .github/workflows/ml_pipeline.yml
name: MLOps CI/CD Pipeline for ML Model

# 當程式碼推送到 main 分支時，觸發這個工作流程
on:
  push:
    branches:
      - main

# 定義一個或多個工作 (Jobs)，每個工作可以包含多個步驟
jobs:
  build-and-train:
    # 這個工作將在最新的 Ubuntu 虛擬機上運行
    runs-on: ubuntu-latest

    # 每個工作由一系列的步驟 (steps) 組成
    steps:
    - name: 🚀 檢查程式碼 (Checkout code)
      # 使用官方 actions/checkout@v3 action 來將程式碼庫複製到虛擬機
      uses: actions/checkout@v3

    - name: 🐍 設定 Python 環境
      # 使用官方 actions/setup-python@v4 action 來設定 Python 環境
      uses: actions/setup-python@v4
      with:
        python-version: '3.9' # 指定你需要的 Python 版本

    - name: ⚙️ 安裝依賴套件
      # 運行 Shell 命令來安裝 requirements.txt 中列出的所有套件
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: 🧪 運行單元測試
      # 運行 pytest 來執行 test_model.py 中的測試
      run: pytest test_model.py

    - name: 🤖 訓練並儲存模型
      # 運行 model.py 腳本來訓練模型並將其保存
      run: python model.py

    - name: ✨ 上傳訓練好的模型及預測結果 (作為 Artifact)
      # 使用官方 actions/upload-artifact@v3 action 來上傳指定的文件
      uses: actions/upload-artifact@v3
      with:
        name: trained-model-and-predictions # artifact 的名稱
        path: |
          model.pkl          # 上傳訓練好的模型檔案
          predictions.txt    # 上傳預測輸出檔案
        # retention-days: 5 # 可以設定 artifact 保存天數，預設是 90 天
```

#### 步驟三：推送到 GitHub 並觀察結果

1.  將所有這些檔案添加到你的 Git 倉庫：
    ```bash
    git add .
    git commit -m "feat: Add MLOps CI/CD pipeline and model training script"
    ```
2.  將這些變更推送到你的 GitHub 倉庫的 `main` 分支：
    ```bash
    git push origin main
    ```

3.  **觀察 GitHub Actions 的運行：**
    *   打開你的 GitHub 倉庫頁面。
    *   點擊頂部的 "Actions" 選項卡。
    *   你應該會看到一個正在運行的工作流程，其名稱為 "MLOps CI/CD Pipeline for ML Model"。
    *   點擊這個工作流程，你可以看到每個步驟的詳細日誌輸出。
    *   當所有步驟都成功完成後，你會看到一個綠色的勾勾！👍
    *   在工作流程的運行頁面，你還會看到一個 "Artifacts" 區塊，點擊你可以下載 `trained-model-and-predictions` 這個壓縮檔，裡面就包含了你的 `model.pkl` 和 `predictions.txt`。

---

### 💡 程式碼解析與意義

*   **`on: push`**: 這個關鍵字定義了什麼時候觸發你的工作流程。這裡設定為當有程式碼推送到 `main` 分支時。
*   **`runs-on: ubuntu-latest`**: 指定你的工作流程將在哪種操作系統的虛擬機上運行。
*   **`uses: actions/checkout@v3`**: 這是一個預設的 GitHub Action，用於將你的程式碼從倉庫複製到虛擬機上，讓後續的步驟可以存取這些檔案。
*   **`uses: actions/setup-python@v4`**: 另一個預設 Action，用於在虛擬機上設定指定版本的 Python 環境。
*   **`run: pip install -r requirements.txt`**: 執行一個 Shell 命令，安裝所有模型所需的依賴套件。這確保了每次訓練的環境都是一致的。
*   **`run: pytest test_model.py`**: 執行你的測試腳本。在真實專案中，這裡會有嚴格的單元測試、整合測試來驗證程式碼邏輯、數據處理和模型基本功能。
*   **`run: python model.py`**: 執行你的模型訓練腳本。這就是自動化訓練的核心！
*   **`uses: actions/upload-artifact@v3`**: 這個 Action 非常重要！它允許你將工作流程中生成的任何檔案（例如：訓練好的模型、訓練報告、預測結果）上傳並儲存為 "artifacts"。這些 artifacts 可以被下載，用於後續的部署、監控或分析。這也是實現模型版本管理的第一步。

### 🚀 接下來你可以做什麼？

今天的範例只是一個簡單的開始。在真實的 MLOps 專案中，你的 CI/CD 流水線會更加複雜和強大：

*   **數據版本控制 (Data Version Control, DVC)**：確保每次訓練都使用特定版本的數據。
*   **模型評估與比較**：自動運行更詳細的模型評估，並將結果記錄到模型註冊中心（例如 MLflow）。
*   **模型註冊中心 (Model Registry)**：將訓練好的模型註冊到一個集中式服務，方便管理不同版本的模型。
*   **自動化部署**：如果模型表現達到預期，自動將模型打包成 Docker 映像，並部署到雲端服務（如 AWS SageMaker、Google AI Platform、Azure Machine Learning）或 Kubernetes 集群。
*   **模型監控 (Model Monitoring)**：部署後持續監控模型的表現，當模型性能下降時自動發出警報。

---

### 總結

太棒了！你今天不僅學會了 MLOps 和 CI/CD 的基本概念，還親手在 GitHub 上搭建了一個自動化模型訓練和打包的流程。這可是把你的機器學習專案從「程式碼」轉變為「產品」的關鍵一步！

雖然這只是個起點，但你已經掌握了 MLOps 自動化的核心思想。從現在開始，你可以想像一下，未來你的模型只要一推程式碼，就能自動測試、自動訓練、自動部署，是不是感覺超級酷炫又專業呢？

繼續保持好奇心，不斷探索 MLOps 的更多可能性吧！期待你在這個領域發光發熱！下個階段見！ 💪