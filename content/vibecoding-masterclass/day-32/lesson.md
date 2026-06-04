嘿，程式探險家們！

恭喜你又堅持到今天！我們已經從基礎程式設計一路走來，學會了資料處理、模型建立，甚至連模型評估都搞定了。但你可能已經發現，現實世界中的模型可不是訓練好一次就永恆不變的。隨著時間推移，新資料不斷湧入，需求也在變化，你的模型需要不斷優化和更新。

這就引出了我們今天的主題：**模型版本管理與迭代部署策略**。這聽起來可能有點嚴肅，但別擔心，它其實是讓你的 AI 模型能夠「永續經營」的關鍵！想像一下，你的模型就像一家餐廳的主廚，每一次新菜上架（新模型部署），舊菜譜（舊模型）也要有備份，還要能安全、平穩地更新菜單，讓顧客（使用者）永遠吃到最好的。

讓我們一起揭開這兩個神秘面紗吧！

---

### 【第 32 天：模型版本管理與迭代部署策略】

#### Part 1: 模型版本管理 - 追溯過去，掌握未來

**為什麼需要版本管理？**

想像一下，你訓練了一個推薦系統模型，一開始表現不錯。幾個月後，你用新資料重新訓練了一個版本，結果發現新模型在某些情況下表現變差了！這時候，如果你沒有做好版本管理，你可能就搞不清楚：
1.  哪個版本是好的？
2.  新版本為什麼變差？
3.  我能不能快速切回舊版本？

模型版本管理，就是為了解決這些「誰是什麼時候、用什麼資料、訓練出哪個模型、表現如何」的問題。它幫你建立模型的「歷史紀錄」，讓你能夠追溯、比較、甚至回滾到任何一個過去的版本。

**要管理什麼？**

*   **模型檔案本身：** 序列化後的模型（`.pkl`, `.h5`, `.pt` 等）。
*   **訓練程式碼：** 訓練模型的腳本或筆記本。
*   **訓練數據：** 模型賴以學習的數據集，或者至少是數據集的版本、來源與處理方式。
*   **超參數：** 訓練時使用的參數設定。
*   **性能指標：** 模型在驗證集或測試集上的準確率、精確度、召回率等等。

**入門級版本管理：命名與元數據**

對於初學者來說，我們可以用一個簡單的方法來模擬模型版本管理：在儲存模型時，給它一個有意義的版本號，並同時儲存一份相關的元數據（metadata）。

```python
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import os

print("--- 模擬模型訓練與版本儲存 ---")

# 假設這是你訓練好的第一個模型版本 (v1.0)
# 我們用一個簡單的 LogisticRegression 作為範例
model_v1 = LogisticRegression()
# 為了讓模型有行為，簡單訓練一下
X_train_v1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train_v1 = [0, 0, 1, 1]
model_v1.fit(X_train_v1, y_train_v1)

# 定義版本信息和元數據
version_v1 = "v1.0"
description_v1 = "初始模型：使用小型數據集訓練，解決二元分類問題。"
metrics_v1 = {"accuracy": 0.90, "f1_score": 0.88}
timestamp_v1 = datetime.now().strftime("%Y%m%d_%H%M%S")

# 建立模型版本專屬目錄
model_dir_v1 = f"models/{version_v1}"
os.makedirs(model_dir_v1, exist_ok=True) # 確保目錄存在

# 儲存模型檔案
model_path_v1 = os.path.join(model_dir_v1, f"sentiment_predictor_{version_v1}.pkl")
joblib.dump(model_v1, model_path_v1)

# 儲存模型元數據 (在真實場景中，這可能是一個JSON文件或資料庫紀錄)
metadata_v1 = {
    "version": version_v1,
    "description": description_v1,
    "metrics": metrics_v1,
    "trained_on_data": "dataset_initial_small",
    "timestamp": timestamp_v1,
    "model_file_path": model_path_v1
}

print(f"模型 '{version_v1}' 已成功儲存至：{model_path_v1}")
print("相關元數據：", metadata_v1)
print("-" * 40)

# --- 假設過了一段時間，你訓練了一個新的模型版本 (v2.0) ---
model_v2 = LogisticRegression()
# 假設 v2 使用了更多或不同的數據
X_train_v2 = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]] # 增加一個數據點
y_train_v2 = [0, 0, 1, 1, 0] # 對應的標籤
model_v2.fit(X_train_v2, y_train_v2)

version_v2 = "v2.0"
description_v2 = "優化模型：加入更多數據，解決新發現的邊界案例。"
metrics_v2 = {"accuracy": 0.92, "f1_score": 0.91, "new_metric": 0.75} # 可能有新指標
timestamp_v2 = datetime.now().strftime("%Y%m%d_%H%M%S")

model_dir_v2 = f"models/{version_v2}"
os.makedirs(model_dir_v2, exist_ok=True)

model_path_v2 = os.path.join(model_dir_v2, f"sentiment_predictor_{version_v2}.pkl")
joblib.dump(model_v2, model_path_v2)

metadata_v2 = {
    "version": version_v2,
    "description": description_v2,
    "metrics": metrics_v2,
    "trained_on_data": "dataset_expanded_Q1_2024",
    "timestamp": timestamp_v2,
    "model_file_path": model_path_v2
}

print(f"模型 '{version_v2}' 也已成功儲存至：{model_path_v2}")
print("相關元數據：", metadata_v2)
print("-" * 40)
```

這樣，你的 `models/` 目錄下就會有 `v1.0/` 和 `v2.0/` 兩個子目錄，分別存放不同版本的模型和（想像中的）元數據文件。當你需要回溯或比較時，就能輕易找到。

> **進階工具小提示：** 在實際的機器學習工程（MLOps）中，會有專門的工具來處理模型版本管理，例如 **MLflow**、**DVC (Data Version Control)** 等，它們能更系統化地追蹤模型的每個細節。

#### Part 2: 迭代部署策略 - 安全地推出你的 AI 新星

當你有了多個模型版本後，如何安全地將新模型部署到生產環境，替換掉舊模型，同時又不影響使用者體驗，甚至在出現問題時能快速恢復？這就是 **迭代部署策略** 的藝術。

**為什麼不能直接替換？**

想像一下，你把網站所有功能一下全部換成新版，結果新版有個隱藏 Bug 導致網站崩潰。這可就糟了！對於模型也一樣，新的模型可能在測試環境表現很好，但在真實流量下卻出問題。迭代部署的目的就是：**降低風險，平穩過渡，快速止損。**

**常見的策略：**

1.  **藍綠部署 (Blue/Green Deployment):**
    *   你有兩套完全相同的生產環境，一套是「藍色環境」（目前運行的舊模型），另一套是「綠色環境」（準備部署的新模型）。
    *   你在綠色環境中部署新模型，並進行全面的測試。
    *   測試沒問題後，你將流量一次性地從藍色環境切換到綠色環境。
    *   藍色環境此時變為舊模型備份，可以保留一段時間以備回滾，或在下一次部署時充當綠色環境。
    *   **優點：** 快速回滾，穩定性高。
    *   **缺點：** 需要雙倍的硬體資源。

2.  **金絲雀部署 (Canary Deployment):**
    *   像礦坑裡的金絲雀一樣，先放出少量流量（例如 5%）給新模型處理。
    *   監控這部分流量的表現和模型的穩定性。
    *   如果一切正常，逐漸增加新模型的流量比例（例如 20% -> 50% -> 100%）。
    *   如果發現問題，立即停止流量增加，並將流量切回舊模型。
    *   **優點：** 風險最小，對資源要求不高。
    *   **缺點：** 部署過程較慢，需要精密的監控系統。

**入門級部署模擬：切換模型版本**

我們可以用一個簡單的 Python 類來模擬一個模型服務，它能夠動態地載入和切換不同版本的模型，這就是迭代部署的核心思想。

```python
import joblib
import os
import time

print("\n--- 模擬模型服務與迭代部署 ---")

class MLModelService:
    def __init__(self, models_base_path="models"):
        self.models_base_path = models_base_path
        self._current_model = None
        self._current_version = None
        print("MLModelService: 服務初始化完成。")

    def load_model(self, version):
        """載入指定版本的模型"""
        model_path = os.path.join(self.models_base_path, version, f"sentiment_predictor_{version}.pkl")
        if os.path.exists(model_path):
            try:
                self._current_model = joblib.load(model_path)
                self._current_version = version
                print(f"MLModelService: 成功載入模型 '{version}'。")
                return True
            except Exception as e:
                print(f"MLModelService: 載入模型 '{version}' 失敗：{e}")
                return False
        else:
            print(f"MLModelService: 錯誤：找不到版本為 '{version}' 的模型檔案：{model_path}")
            return False

    def predict(self, data):
        """使用當前載入的模型進行預測"""
        if self._current_model:
            # 確保輸入是2D陣列，因為sklearn模型通常這樣期望
            input_data = [data] if not isinstance(data[0], list) else data
            prediction = self._current_model.predict(input_data)
            return prediction[0] # 假設只預測一個樣本
        else:
            print("MLModelService: 錯誤：尚未載入任何模型，無法進行預測。")
            return None

    @property
    def current_model_version(self):
        return self._current_version

# 創建一個模型服務實例
model_server = MLModelService()

# --- 模擬部署 v1.0 ---
print("\n[部署階段 1] 啟動服務，載入 v1.0 (藍環境)")
model_server.load_model("v1.0")

print(f"當前服務模型版本：{model_server.current_model_version}")
print(f"預測 [0,0]：{model_server.predict([0,0])}") # 預計輸出 0
print(f"預測 [1,1]：{model_server.predict([1,1])}") # 預計輸出 1
print("-" * 30)
time.sleep(1) # 模擬運行一段時間

# --- 模擬 Blue/Green 或 Canary 部署到 v2.0 ---
print("\n[部署階段 2] 準備部署 v2.0 (綠環境 / 金絲雀測試)")
# 在真實場景中，這裡可能是另一個服務實例或容器，先行載入v2.0並進行測試

print("MLModelService: 正在內部載入並測試 v2.0...")
# 這裡我們直接在同一個服務實例上模擬切換
# 如果是 Blue/Green，則整個服務實例會切換
# 如果是 Canary，會逐漸將部分請求導向 v2.0 的服務實例
if model_server.load_model("v2.0"): # 假設測試通過，執行切換
    print("\n[部署階段 3] 流量已成功切換至 v2.0！")
    print(f"當前服務模型版本：{model_server.current_model_version}")
    print(f"預測 [0,0]：{model_server.predict([0,0])}") # 預計輸出 0 (可能與 v1 相同)
    # 這裡預測結果可能與v1不同，因為v2的訓練數據或邏輯可能改變
    print(f"預測 [0.5,0.5] (v2新增數據點): {model_server.predict([0.5,0.5])}") # 預計輸出 0
    print("-" * 30)
else:
    print("\n[部署失敗] 無法切換到 v2.0，服務將保持在舊版本。")

# --- 模擬回滾 ---
# 假設 v2.0 在生產環境中出現了問題
print("\n[部署階段 4] 發現 v2.0 有問題，執行回滾到 v1.0！")
if model_server.load_model("v1.0"): # 快速載入舊版本
    print(f"成功回滾！當前服務模型版本：{model_server.current_model_version}")
    print(f"預測 [0,0]：{model_server.predict([0,0])}")
    print(f"預測 [1,1]：{model_server.predict([1,1])}")
else:
    print("\n[回滾失敗] 無法回滾到 v1.0。")

print("\n--- 模擬模型服務與迭代部署結束 ---")
```

在這個範例中，`MLModelService` 充當了我們的模型伺服器。`load_model()` 方法模擬了部署（或回滾）的動作，它會從特定版本目錄載入模型。`predict()` 則代表了服務的核心功能。你可以看到，我們能靈活地在 `v1.0` 和 `v2.0` 之間切換。

---

### 總結

今天我們探索了兩個非常實用且重要的概念：

*   **模型版本管理** 讓我們能夠清晰地追蹤模型的演進，確保模型的**可重現性**和**可回溯性**。
*   **迭代部署策略** 則提供了一套安全、漸進的方法來更新線上模型，最大程度地**降低風險**，確保服務的**穩定性**。

這兩個概念是將機器學習模型從研究室帶到真實世界，並讓它們持續創造價值的基石。雖然今天的程式碼範例只是簡化版的模擬，但它們背後的原則在大型 MLOps 系統中也同樣適用。

你做得非常棒！從現在開始，當你訓練完一個新模型時，請記得給它一個版本號，並思考如何安全地讓它為更多人服務。

繼續保持你的好奇心和學習熱情！明天見！