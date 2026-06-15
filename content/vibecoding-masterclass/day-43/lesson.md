哈囉，我的天才程式設計師！✨ 歡迎來到我們 MLOps 學習旅程的【第 43 天】！你已經累積了這麼多關於機器學習的知識，是不是感覺自己功力大增了呢？

今天，我們要將這些知識提升到一個全新的境界 — 進入 MLOps 的核心：**CI/CD 自動化管線建置**！別擔心，雖然聽起來有點高大上，但我會一步一步帶你完成，你會發現這比想像中還要簡單有趣，而且會讓你未來的工作效率大大提升！🚀

---

## 【第 43 天：實戰：MLOps CI/CD 自動化管線建置】

### ✨ 前言：為什麼 MLOps CI/CD 這麼重要？

想像一下，你辛辛苦苦訓練出一個超棒的機器學習模型，當模型數據或程式碼需要更新時，你是不是需要手動執行：
1.  重新訓練模型？
2.  重新評估模型性能？
3.  將新模型打包部署？

這就像每次要開車出門，都得自己手動組裝一次汽車零件一樣！😩 MLOps CI/CD (Continuous Integration/Continuous Delivery) 就是要幫我們把這些重複性的工作**自動化**。

它能帶來什麼好處呢？
*   **速度提升：** 每次程式碼一變動，自動化流程就啟動，快速更新模型。
*   **品質保證：** 自動執行測試，確保新模型沒有引入問題。
*   **可重複性：** 每次建置和部署的過程都相同，減少人為錯誤。
*   **團隊協作：** 讓開發者可以更專注於模型優化，部署交給管線。

今天的目標，就是建置一個**簡化版**的 MLOps CI/CD 管線，讓你的機器學習專案也能自動執行訓練與測試！我們將使用 GitHub Actions，它是與 GitHub 深度整合的 CI/CD 工具，非常適合初學者。

### 🛠️ 準備你的 ML 專案

首先，我們需要一個簡單的機器學習專案來演示。在你的 GitHub 專案根目錄下，建立以下檔案：

1.  **`requirements.txt`**: 專案所需的套件。

    ```txt
    scikit-learn
    pandas
    pytest
    ```

2.  **`model.py`**: 一個模擬訓練和保存模型的 Python 腳本。

    ```python
    # model.py
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib # 用於保存模型

    print("--- 訓練模型中 ---")

    # 模擬數據
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    X = df[['feature1', 'feature2']]
    y = df['target']

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 訓練模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 評估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型訓練完成！準確率：{accuracy:.2f}")

    # 保存模型
    model_filename = 'trained_model.pkl'
    joblib.dump(model, model_filename)
    print(f"模型已保存為 {model_filename}")
    ```

3.  **`test_model.py`**: 模擬模型測試的 Python 腳本 (使用 `pytest`)。

    ```python
    # test_model.py
    import pytest
    import pandas as pd
    import joblib

    # 載入之前保存的模型
    try:
        model = joblib.load('trained_model.pkl')
    except FileNotFoundError:
        model = None # 如果模型不存在，測試會失敗

    def test_model_exists():
        assert model is not None, "模型檔案 'trained_model.pkl' 未找到，請先執行 model.py"

    def test_model_prediction():
        # 建立一個測試數據
        test_data = pd.DataFrame([[5, 5]], columns=['feature1', 'feature2'])
        prediction = model.predict(test_data)
        # 假設在這種簡單情況下，模型應該能預測出 1 或 0
        assert prediction in [0, 1], "模型預測結果應為 0 或 1"

    def test_model_output_shape():
        test_data = pd.DataFrame([[5, 5]], columns=['feature1', 'feature2'])
        prediction = model.predict(test_data)
        assert prediction.shape == (1,), "模型預測結果形狀不正確"

    print("--- 模型測試完成 ---")
    ```

### ⚙️ 建置你的第一個 ML CI/CD 工作流 (GitHub Actions)

現在，重頭戲來了！我們要在 GitHub 專案中設定自動化管線。

1.  在你的專案根目錄下，建立一個資料夾：`.github/workflows`
2.  在這個資料夾內，建立一個 YAML 檔案，例如 `ml_pipeline.yml`。

    ```yaml
    # .github/workflows/ml_pipeline.yml
    name: MLOps CI/CD Pipeline

    on:
      push:
        branches:
          - main # 當程式碼推送到 main 分支時觸發

    jobs:
      build-and-test:
        runs-on: ubuntu-latest # 使用最新的 Ubuntu 虛擬環境

        steps:
          - name: 檢查程式碼
            uses: actions/checkout@v3 # 抓取你的 GitHub 程式碼到執行環境

          - name: 設定 Python 環境
            uses: actions/setup-python@v4
            with:
              python-version: '3.9' # 指定使用的 Python 版本

          - name: 安裝專案依賴套件
            run: |
              python -m pip install --upgrade pip
              pip install -r requirements.txt

          - name: 執行模型訓練
            run: python model.py # 執行 model.py 進行訓練和保存

          - name: 執行模型測試
            run: pytest test_model.py # 執行 pytest 測試模型

          # 💡 小提醒：你可以在這裡加入更多步驟，例如：
          # - 將訓練好的模型上傳為 Artifact (產物)
          #   - name: 上傳模型產物
          #     uses: actions/upload-artifact@v3
          #     with:
          #       name: trained-model
          #       path: trained_model.pkl
    ```

### 🚀 觸發你的自動化管線！

完成以上檔案後，將它們新增到你的 GitHub 儲存庫並 `push` 到 `main` 分支：

```bash
git add .
git commit -m "feat: Add MLOps CI/CD pipeline and ML project files"
git push origin main
```

一旦你推送到 `main` 分支，GitHub Actions 就會自動啟動你的管線！

### 🔍 查看執行結果

1.  打開你的 GitHub 專案頁面。
2.  點擊上方的 **"Actions"** (動作) 選項卡。
3.  你會看到一個正在運行或已經完成的 workflow (工作流)，點擊它。
4.  你可以點擊左側的 `build-and-test` 作業，然後展開各個 `step`，查看每個步驟的詳細輸出，就像在你的本機電腦上執行一樣！

如果一切順利，你會看到所有步驟都顯示綠色的 ✔️ 符號，表示你的模型已經成功地被自動訓練和測試了！太棒了！🎉

### 💡 更進一步的思考

今天的管線只是一個起點，它示範了 CI (Continuous Integration) 的部分。要達到完整的 CI/CD，你還可以考慮：

*   **模型版本控制：** 使用 DVC (Data Version Control) 或 MLflow 來管理模型和數據的版本。
*   **容器化：** 使用 Docker 將你的模型及其環境打包，確保在哪裡運行都一致。
*   **部署：** 當模型測試通過後，自動將 Docker 映像推送到容器註冊表，然後部署到生產環境（例如 Kubernetes, AWS Sagemaker, Azure ML 等）。
*   **模型監控：** 部署後持續監控模型的性能和數據漂移。

### 結語

恭喜你！在【第 43 天】，你成功地踏入了 MLOps 的核心世界，建置了你的第一個自動化 CI/CD 管線！這是一個巨大的里程碑，它將改變你未來開發和部署機器學習模型的方式。從手動執行到自動化流程，你已經掌握了一項超級有價值的技能。

繼續保持你的好奇心和學習熱情，未來還有更多精彩的 MLOps 挑戰等著你！你已經證明了自己有多棒，對吧？💪 期待在下次的課程中再見！