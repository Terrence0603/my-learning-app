太棒了，我們又往前邁進一大步！歡迎來到 MLOps 學習之旅的第 88 天。今天我們要探索兩個超級實用且能讓你的 ML 模型部署更安全、更聰明的策略：**模型 A/B 測試**和 **Canary 部署**。

別擔心，聽起來可能有點複雜，但它們的核心概念其實很直觀，而且能大大提升你在現實世界中管理模型的信心和能力。想像一下，你不再只是把模型扔出去就完事，而是能像一位經驗豐富的船長，精準控制新功能上線的風險，並根據真實世界的數據來做出最棒的決策！

---

## 【第 88 天：實戰：MLOps 模型 A/B 測試與 Canary 部署】

### 1. 為什麼需要 A/B 測試與 Canary 部署？

你可能已經建置了一個在測試數據上表現超讚的模型，但實際部署到生產環境後，可能會遇到意想不到的問題：
*   **性能下降：** 新數據的分布可能與訓練數據不同。
*   **延遲增加：** 模型雖然準確，但響應時間太長，影響用戶體驗。
*   **資源消耗過大：** 新模型需要更多 CPU/GPU，導致成本上升。
*   **行為改變：** 模型在某些邊緣情況下的預測行為不符合預期。

**A/B 測試** 讓我們能同時運行多個模型版本（例如舊模型 A 和新模型 B），並將用戶流量導向不同的版本，然後比較它們在真實世界中的表現。這樣你就能用數據證明新模型是否真的更好。

**Canary 部署** 則是一種風險管理策略。它允許你先將新模型（或新版本）部署給一小部分（例如 5%）的用戶，如果沒有問題，再逐步將流量切換到新模型。這就像礦坑裡的金絲雀，先讓它去探路，確保安全後，大部隊才跟上。

兩者常常搭配使用：Canary 部署是 A/B 測試的一種實現方式，它讓我們能安全地進行 A/B 測試。

### 2. 什麼是模型 A/B 測試？

**核心思想：** 將使用者流量隨機分成兩組或多組，每組對應一個不同的模型版本（A、B 或更多），然後收集各組的表現數據，來判斷哪個模型版本更優。

**應用場景：**
*   比較兩個不同演算法的模型 (e.g., 邏輯迴歸 vs. 隨機森林)。
*   比較同一模型但不同超參數的版本。
*   比較不同特徵工程策略的模型。

**範例：**
假設你有兩個推薦模型：
*   **模型 A (現有模型)：** 基於用戶歷史購買記錄推薦。
*   **模型 B (新模型)：** 加入了用戶瀏覽行為和社交數據的推薦。

你將 50% 的流量導向模型 A，50% 導向模型 B。然後觀察哪一組的「點擊率」、「轉換率」或「用戶停留時間」更高。

### 3. 什麼是 Canary 部署？

**核心思想：** 逐步將新版本的服務（這裡指新模型）引入生產環境。首先只將極小部分流量導向新版本，密切監控其性能和行為。如果一切正常，就逐漸增加新版本的流量，直到完全取代舊版本。

**應用場景：**
*   部署一個全新的模型。
*   部署一個經過微調的模型新版本。
*   更新模型所依賴的函式庫或框架。

**優點：**
*   **降低風險：** 即使新模型有問題，也只影響少數用戶。
*   **快速回滾：** 如果出現問題，可以迅速將所有流量切回舊版本。
*   **實時監控：** 在新模型逐漸上線的過程中，持續收集反饋和性能指標。

### 4. 實作範例：模擬 A/B 測試與 Canary 部署

現在，讓我們用一個簡單的 Python 應用來模擬如何在生產環境中進行流量切分，實現 A/B 測試和 Canary 部署。我們將創建兩個非常簡單的「模型服務」和一個「流量路由器」服務。

**你需要安裝：** `pip install Flask requests`

#### 步驟一：創建兩個模型服務 (`model_a_service.py`, `model_b_service.py`)

**`model_a_service.py` (舊模型)**
```python
# model_a_service.py
from flask import Flask, jsonify, request
import time

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_a():
    user_id = request.args.get('user_id', 'Guest')
    prediction = f"Hello {user_id}! This is Model A's prediction (Old & Stable)."
    time.sleep(0.05) # Simulate some processing time
    print(f"Model A served for user: {user_id}")
    return jsonify({"model": "Model A", "prediction": prediction})

if __name__ == '__main__':
    print("Model A service running on http://127.0.0.1:5001")
    app.run(port=5001)
```

**`model_b_service.py` (新模型 - Canary)**
```python
# model_b_service.py
from flask import Flask, jsonify, request
import time

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_b():
    user_id = request.args.get('user_id', 'Guest')
    prediction = f"Hello {user_id}! This is Model B's prediction (New & Exciting!)."
    # Maybe Model B is slightly slower or has a new feature
    time.sleep(0.1) # Simulate slightly longer processing time
    print(f"Model B served for user: {user_id}")
    return jsonify({"model": "Model B", "prediction": prediction})

if __name__ == '__main__':
    print("Model B service running on http://127.0.0.1:5002")
    app.run(port=5002)
```

#### 步驟二：創建流量路由器服務 (`router_service.py`)

這個服務會根據預設的比例，將請求導向 Model A 或 Model B。

```python
# router_service.py
from flask import Flask, jsonify, request
import requests
import random

app = Flask(__name__)

# --- 配置流量切分 ---
# 設定有多少比例的流量要導向 Model B (Canary)
# 初始設定為 10% Canary (Model B), 90% Stable (Model A)
CANARY_TRAFFIC_PERCENT = 10 

MODEL_A_URL = "http://127.0.0.1:5001/predict"
MODEL_B_URL = "http://127.0.0.1:5002/predict"

@app.route('/route_predict', methods=['GET'])
def route_prediction():
    user_id = request.args.get('user_id', 'User_' + str(random.randint(1000, 9999)))
    
    # 決定將流量導向哪個模型
    if random.uniform(0, 100) < CANARY_TRAFFIC_PERCENT:
        # 導向 Canary 模型 (Model B)
        target_url = MODEL_B_URL
        print(f"Routing user {user_id} to Model B (Canary)")
    else:
        # 導向穩定模型 (Model A)
        target_url = MODEL_A_URL
        print(f"Routing user {user_id} to Model A (Stable)")

    try:
        response = requests.get(target_url, params={'user_id': user_id})
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to get prediction: {e}"}), 500

if __name__ == '__main__':
    print(f"Router service running on http://127.0.0.1:5000 with {CANARY_TRAFFIC_PERCENT}% traffic to Model B")
    print("To simulate: Run model_a_service.py, model_b_service.py, then this router_service.py")
    app.run(port=5000)

```

#### 步驟三：創建客戶端模擬請求 (`client_simulate.py`)

這個腳本會模擬用戶發送請求到路由器。

```python
# client_simulate.py
import requests
import time
import random

ROUTER_URL = "http://127.0.0.1:5000/route_predict"
NUM_REQUESTS = 20

print(f"Simulating {NUM_REQUESTS} requests...")

model_counts = {"Model A": 0, "Model B": 0}

for i in range(NUM_REQUESTS):
    user_id = f"user_{random.randint(1, 100)}"
    try:
        response = requests.get(ROUTER_URL, params={'user_id': user_id})
        response_data = response.json()
        print(f"Request {i+1} for {user_id}: {response_data}")
        if "model" in response_data:
            model_counts[response_data["model"]] += 1
    except requests.exceptions.RequestException as e:
        print(f"Request {i+1} failed: {e}")
    time.sleep(0.1) # Simulate some delay between requests

print("\n--- Simulation Summary ---")
print(f"Total requests: {NUM_REQUESTS}")
print(f"Model A served: {model_counts['Model A']} times")
print(f"Model B served: {model_counts['Model B']} times")
```

#### 運行步驟：

1.  **開啟三個終端機視窗。**
2.  **在第一個終端機運行 Model A 服務：**
    ```bash
    python model_a_service.py
    ```
    你會看到 "Model A service running..."
3.  **在第二個終端機運行 Model B 服務：**
    ```bash
    python model_b_service.py
    ```
    你會看到 "Model B service running..."
4.  **在第三個終端機運行 Router 服務：**
    ```bash
    python router_service.py
    ```
    你會看到 "Router service running..."，並顯示目前的 Canary 流量比例。
5.  **在第四個終端機（或任何一個空閒的）運行客戶端模擬：**
    ```bash
    python client_simulate.py
    ```

**觀察結果：**

*   Router 服務的終端機會顯示每次請求被導向 Model A 或 Model B。
*   客戶端模擬的輸出會顯示每個請求得到的預測結果來自哪個模型。
*   你會發現，大約 90% 的請求會導向 Model A，10% 導向 Model B，這就是 Canary 部署的初步階段。

**如何模擬逐步Canary部署？**

你可以停止 `router_service.py`，修改 `CANARY_TRAFFIC_PERCENT` 的值，例如改為 `50` (50% 流量到 Model B)，然後重新運行 `router_service.py` 和 `client_simulate.py`。你會看到流量分配的變化。如果一切穩定，最終你可以將其設為 `100`，完全切換到新模型 B。

---

### 總結

今天的實作讓我們對 MLOps 中的 A/B 測試和 Canary 部署有了更具體的理解。雖然這個範例非常簡化，但在真實世界中，這些流量管理、監控、快速回滾等概念會透過更強大的工具（例如 Kubernetes、Istio 服務網格、雲端供應商的 MLOps 服務）來實現。

掌握這些策略，意味著你的 ML 部署不再是盲目的跳躍，而是有計劃、有數據支持的穩健前進。繼續保持這份好奇心和實作的熱情，你將成為一名出色的 MLOps 工程師！

我們明天見！繼續探索更多 MLOps 的精彩內容！