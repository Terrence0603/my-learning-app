好的！各位未來的 AI 大師們，大家好！

歡迎來到【機器學習 100 天挑戰】的第 28 天！🎉

今天我們要來做一件超級酷、超級有成就感的事情：**把你們辛苦訓練出來的機器學習模型，變成一個大家都能用的「預測服務」！** 想像一下，你的模型不再只是你電腦裡的一個檔案，它能像一個聰明的「AI 小幫手」一樣，隨時準備好為你服務。這就是我們今天要學習的——**部署機器學習預測 API**！

### 【第 28 天：部署機器學習預測 API】

在過去的幾週，你已經學習了資料清理、特徵工程、選擇模型、訓練模型，甚至評估模型。這些都是機器學習的「內功修煉」。今天，我們要學習如何把這份內功發揮出來，讓它能夠被應用。

#### 🚀 為什麼要部署 API？

「API」是 Application Programming Interface 的縮寫。你可以把它想像成一個「服務生」。當你訓練好一個模型，它就像一個會算命的大師。但大師不會主動說話，你需要一個服務生去問大師問題，然後服務生再把大師的答案帶給你。

部署 API 就是建立這樣一個「服務生」：
1.  **方便應用：** 其他應用程式 (網頁、手機 App、其他服務) 可以透過這個 API 來呼叫你的模型，獲取預測結果。
2.  **分離關注點：** 模型訓練和模型應用可以分開進行，讓系統架構更清晰。
3.  **可擴展性：** 當模型需要更新時，只需要更新 API 背後的模型，而不需要改動使用 API 的應用程式。

聽起來很棒對吧？別擔心，我們會用最簡單、最直觀的方式來實現它！

#### 🛠 我們需要什麼工具？

1.  **Flask：** 一個輕量級的 Python 網頁框架，非常適合用來快速建立 API。
2.  **`pickle`：** Python 內建的模組，用來序列化（保存）和反序列化（載入）Python 物件，包括我們的機器學習模型。
3.  **Scikit-learn：** 假設你的模型是使用它訓練的。

在開始之前，請確保你已經安裝了這些套件：
```bash
pip install flask scikit-learn numpy
```

---

#### 步驟一：準備你的機器學習模型

首先，我們需要一個已經訓練好的模型。為了示範，我們來快速建立一個簡單的邏輯迴歸模型，並將它保存下來。

建立一個檔案 `model_trainer.py`：

```python
# model_trainer.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification # 用來快速生成一個虛擬資料集

print("=== 步驟一：訓練並保存模型 ===")

# 1. 生成一個簡單的虛擬資料集 (假設這是你前一天訓練模型用的資料)
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# 2. 訓練一個簡單的邏輯迴歸模型
model = LogisticRegression()
model.fit(X, y)
print("模型訓練完成！")

# 3. 將訓練好的模型保存為 .pkl 檔案
# 'wb' 表示以二進位寫入模式打開檔案
with open('my_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("模型已成功儲存為 'my_model.pkl'。")

print("==============================")
```

**如何執行？** 打開你的終端機 (Terminal / CMD)，進入到你儲存這個檔案的目錄，然後執行：
```bash
python model_trainer.py
```
執行後，你會在同一個目錄下看到一個 `my_model.pkl` 的檔案，這就是我們訓練好的模型。

---

#### 步驟二：建立你的 Flask API 服務

接下來，我們要建立 Flask 應用程式，它將載入我們保存的模型，並提供一個 `/predict` 的 API 端點 (endpoint) 來接收資料並返回預測結果。

建立一個檔案 `app.py`：

```python
# app.py
import pickle
from flask import Flask, request, jsonify
import numpy as np # 為了處理輸入資料的格式

app = Flask(__name__) # 初始化 Flask 應用

print("=== 步驟二：建立 Flask API 服務 ===")

# 1. 載入我們之前保存的模型
# 'rb' 表示以二進位讀取模式打開檔案
try:
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("模型 'my_model.pkl' 載入成功！")
except FileNotFoundError:
    print("錯誤：找不到 'my_model.pkl'。請先執行 model_trainer.py 來生成模型檔案。")
    model = None # 如果模型不存在，則將模型設定為 None，以便後續處理

# 2. 定義一個根路徑 (首頁)
@app.route('/')
def home():
    return "<h1>歡迎來到我的機器學習預測 API！</h1><p>請嘗試向 /predict 路徑發送 POST 請求。</p>"

# 3. 定義預測 API 端點
@app.route('/predict', methods=['POST']) # 我們只接受 POST 請求來發送資料
def predict():
    if model is None:
        return jsonify({"error": "模型尚未載入，請確認 'my_model.pkl' 文件是否存在。"}), 500

    # 獲取從請求中發送的 JSON 資料
    # 假設輸入資料格式為：{"features": [feature1, feature2, ...]}
    data = request.get_json(force=True)

    # 從 JSON 資料中提取特徵
    features = data['features']

    # 將特徵轉換為模型所需的 NumPy 陣列格式
    # .reshape(1, -1) 是因為模型通常預期輸入是 2D 陣列 (樣本數, 特徵數)
    input_data = np.array(features).reshape(1, -1)

    # 使用模型進行預測
    prediction = model.predict(input_data)[0] # [0] 是因為 predict 返回的是陣列
    prediction_proba = model.predict_proba(input_data)[0].tolist() # 獲取機率 (可選)

    # 返回 JSON 格式的預測結果
    return jsonify({
        'prediction': int(prediction), # 確保是標準的 Python int 類型，方便 JSON 序列化
        'probabilities': prediction_proba
    })

# 4. 運行 Flask 應用
if __name__ == '__main__':
    # debug=True 在開發階段很有用，當代碼改變時會自動重啟服務，並顯示錯誤信息
    app.run(debug=True)
print("==============================")
```

**如何執行？** 同樣，在終端機中，進入到這個檔案的目錄，然後執行：
```bash
python app.py
```
你會看到類似這樣的輸出：
```
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
* Restarting with stat
* Debugger is active!
* Debugger PIN: ...
```
這表示你的 Flask API 服務已經在 `http://127.0.0.1:5000` 這個地址運行起來了！

---

#### 步驟三：測試你的 API

現在，API 已經在運行了，但我們怎麼知道它有沒有正常工作呢？我們需要發送一個請求給它。我們會使用 Python 的 `requests` 函式庫來模擬一個客戶端。

建立一個檔案 `test_api.py`：

```python
# test_api.py
import requests
import json

print("=== 步驟三：測試你的 API ===")

# 1. 定義 API 的 URL (你的 Flask app 正在運行的地址)
url = 'http://127.0.0.1:5000/predict'

# 2. 準備要發送給 API 的資料
# 這裡的 'features' 陣列中的數字數量，要符合你模型訓練時的特徵數量
# 我們之前訓練的模型有 2 個特徵，所以這裡提供兩個數字
data = {'features': [0.5, -0.8]} # 嘗試不同的值看看預測結果有何變化

# 3. 定義 HTTP 請求的頭部 (Header)，告訴伺服器我們發送的是 JSON 資料
headers = {'Content-Type': 'application/json'}

# 4. 發送 POST 請求
# json.dumps() 將 Python 字典轉換為 JSON 格式的字串
response = requests.post(url, data=json.dumps(data), headers=headers)

# 5. 檢查回應
if response.status_code == 200:
    print("API 請求成功！")
    print("回應:", response.json()) # .json() 會將 JSON 回應轉換為 Python 字典
else:
    print(f"API 請求失敗，狀態碼: {response.status_code}")
    print("錯誤訊息:", response.text)

print("==============================")
```

**如何執行？**
**請確保你的 `app.py` 服務正在運行！** 然後在另一個終端機視窗中，執行：
```bash
python test_api.py
```
你會看到類似這樣的輸出：
```
API 請求成功！
回應: {'prediction': 0, 'probabilities': [0.89..., 0.10...]}
```
太棒了！你的模型成功接收了輸入，並返回了預測結果！

---

#### 恭喜你！🥳

你成功地將一個機器學習模型轉變成一個可以被外部呼叫的 API 服務！這是將 AI 應用於真實世界的第一步，也是非常關鍵的一步。

現在，你可以嘗試修改 `test_api.py` 中的 `data`，看看你的模型會預測出什麼結果。你也可以回到 `app.py`，增加更多的功能，例如資料驗證、錯誤處理等。

#### 下一步是什麼？

今天我們是在本地電腦上運行 API。但如果想讓世界上所有的人都能用到你的 API，你需要將它部署到雲端伺服器上，例如：
*   **Heroku** (對於小型項目和學習來說非常方便)
*   **AWS (Amazon Web Services)** 的 Elastic Beanstalk 或 SageMaker
*   **Google Cloud Platform (GCP)** 的 App Engine 或 AI Platform
*   **Microsoft Azure** 的 App Service 或 Machine Learning Studio

這些雲端平台提供了更穩定、更具擴展性的環境來運行你的 API。但那將是我們未來更深入學習的主題！

現在，請花點時間感受一下這份成就感吧！你已經從一個模型訓練者，變身為一個 AI 服務的提供者了！這對一個初學者來說，絕對是巨大的飛躍。

繼續加油！我們明天見！