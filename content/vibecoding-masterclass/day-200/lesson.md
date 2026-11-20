嗨，各位未來的 MLOps 大師！🎉

恭喜你！走到了學習程式設計的第 200 天！這是一段不短的旅程，你可能已經完成了許多很棒的專案，甚至訓練出了一些令人驚豔的機器學習模型。今天，我們要來一場大躍進，把你的模型從「筆記本裡的英雄」變成「真實世界的超能力」！

這就是我們今天的主題：**【第 200 天：MLOps 整合實戰：從模型開發到維運】**

---

### MLOps：為什麼它會是你的下一個超級能力？

你可能已經很熟悉如何準備資料、訓練模型、評估效能。但你有沒有想過：
*   如果你的模型表現很好，要怎麼讓別人也能輕鬆使用它？
*   如果模型過了幾個月效果變差了，要怎麼知道？又該怎麼更新它？
*   如果團隊裡有十幾個人都在訓練模型，大家怎麼知道哪一個是最好的？
*   每次訓練的參數、資料、程式碼變動，要怎麼追蹤？

這些問題的答案，就是 **MLOps (Machine Learning Operations)**！

你可以把 MLOps 想像成是「把你的機器學習模型，從實驗室安全、高效地送上餐桌，並確保它一直美味可靠」的整套流程。它融合了機器學習、DevOps 和資料工程，目標是讓 ML 模型的生命週期更順暢、更可靠、更具擴展性。

別擔心，聽起來很複雜，但我們可以從最核心、最實用的部分開始。

### MLOps 的基石：實驗追蹤與模型版本管理

想像一下，你在廚房裡嘗試了十幾種食譜，但沒有記錄每一種的材料和步驟。下次要做出一樣的菜，或是要改進，就會非常困難！機器學習也是一樣。

這就是 **MLflow** 登場的時候了！MLflow 是一個開源平台，專為機器學習的生命週期管理而設計。它有幾個核心功能，今天我們主要會用到：
1.  **MLflow Tracking：** 記錄你的實驗參數、指標和模型本身。
2.  **MLflow Model Registry：** 管理你的模型版本，讓你知道哪個模型在哪個階段（開發、測試、生產）。

### 實戰演練：從訓練到部署的第一次 MLOps 體驗！

這次我們就用一個簡單的 Scikit-learn 模型來示範如何整合 MLflow。

**步驟一：環境準備**

首先，請確保你已經安裝了必要的函式庫：
```bash
pip install scikit-learn mlflow pandas
```

**步驟二：撰寫訓練與追蹤程式碼**

我們來寫一個 Python 腳本 `train_and_log.py`。這個腳本會訓練一個簡單的分類模型，並使用 MLflow 追蹤實驗結果，最後將模型註冊到 Model Registry。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine

print("--- 開始 ML 訓練與 MLOps 整合 ---")

# 設定 MLflow 實驗名稱
# 這就像給你的廚房實驗一個專屬的筆記本！
mlflow.set_experiment("Wine_Quality_Prediction")

# 載入資料集
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 使用 with mlflow.start_run(): 包裹你的訓練流程
# 這樣 MLflow 就會自動幫你記錄這個實驗的資訊
with mlflow.start_run():
    # 定義模型參數
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # 紀錄模型參數
    # 這些參數會被存在 MLflow Tracking UI 中
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # 初始化並訓練模型
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)

    # 進行預測
    y_pred = model.predict(X_test)

    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 紀錄評估指標
    # 這些指標也會被存在 MLflow Tracking UI 中
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # 將模型儲存到 MLflow Tracking 伺服器
    # 這裡會儲存模型本身，並且記錄模型的相關資訊
    # 'artifact_path' 是模型在 MLflow 儲存路徑下的子路徑
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        registered_model_name="WineQualityPredictor", # 這裡可以指定模型名稱，用於 Model Registry
    )

    print(f"模型已記錄並註冊為 WineQualityPredictor 的一個新版本！")
    print("--- ML 訓練與 MLOps 整合完成 ---")

```

**步驟三：執行訓練腳本並啟動 MLflow UI**

1.  **執行訓練腳本：**
    ```bash
    python train_and_log.py
    ```
    當你執行這個腳本後，它會在背景將實驗結果儲存起來。預設情況下，MLflow 會在你的專案目錄下創建一個 `mlruns` 資料夾來存放這些資訊。

2.  **啟動 MLflow UI：**
    打開你的終端機，切換到你的專案目錄下，然後輸入：
    ```bash
    mlflow ui
    ```
    接著，打開你的瀏覽器，輸入 `http://localhost:5000`。你將會看到一個漂亮的介面，列出了你剛才執行的實驗，包括參數、指標，甚至可以直接下載你的模型！

    在左側選單中，點擊 `Models`，你會看到你的 `WineQualityPredictor` 模型已經被註冊，並且有了第一個版本！這就是你的模型倉庫！

**步驟四：部署模型 (局部測試)**

MLflow 不僅能幫你管理模型，還能快速地將模型部署起來進行測試。我們可以使用 MLflow Model Serving 功能，將你剛剛註冊的模型部署到一個本地 API 服務。

首先，你需要知道你的模型在 Model Registry 中的名稱和版本。假設你剛註冊的版本是 `Version 1`，並且它的 Stage 是 `None` (預設)。為了方便，我們先把它轉為 `Staging` 或 `Production` 階段：

1.  回到 MLflow UI (`http://localhost:5000`)。
2.  點擊左側的 `Models`。
3.  點擊 `WineQualityPredictor`。
4.  點擊 `Version 1`。
5.  在 `Stage` 旁邊，點擊 `None`，然後選擇 `Transition to -> Staging` (或 Production)。這樣，我們就可以指定部署特定階段的模型。

現在，在你的終端機中（開一個新的，因為 `mlflow ui` 佔用一個），執行部署指令：
```bash
mlflow models serve -m "models:/WineQualityPredictor/Staging" --port 5001
```
這個指令會將 `WineQualityPredictor` 模型中處於 `Staging` 階段的最新版本部署到 `http://localhost:5001`。

**步驟五：測試部署的模型**

模型部署好後，我們就可以發送 HTTP 請求來測試它。你需要準備一些資料作為輸入。

開另一個終端機，執行以下 `curl` 指令：

```bash
# 假設我們要預測一組新的葡萄酒資料
# 這些特徵值需要與你的模型訓練時的輸入特徵順序和數量一致
curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{
    "dataframe_split": {
        "columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
        "data": [
            [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050],
            [12.37, 1.07, 2.1, 18.5, 88, 3.52, 3.75, 0.24, 1.95, 4.5, 1.04, 2.77, 660]
        ]
    }
}'
```
你將會收到模型的預測結果！這代表你的模型已經成功地從實驗室走到了服務台！

### 恭喜你，MLOps 初體驗成功！

這只是 MLOps 的冰山一角！今天你學會了：
*   使用 **MLflow Tracking** 追蹤你的實驗細節。
*   使用 **MLflow Model Registry** 管理你的模型版本和生命週期階段。
*   將註冊的模型部署到本地服務進行 **測試**。

未來，你可以進一步學習如何整合 CI/CD (持續整合/持續部署) 工具，讓模型訓練、測試和部署自動化；學習如何使用 Docker 和 Kubernetes 進行大規模部署；以及如何建立完善的監控系統，確保模型在生產環境中持續表現良好。

最重要的是，你已經跨出了關鍵的第一步！從現在開始，你的機器學習模型將不再只是筆記本裡的數字，而是可以被管理、被部署、被使用的真實世界超能力！繼續探索，保持熱情，未來屬於你！💪✨