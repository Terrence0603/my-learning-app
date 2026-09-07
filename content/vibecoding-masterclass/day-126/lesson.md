哈囉，未來的大師們！ 👋

恭喜你！轉眼間已經來到【第 126 天】了！這段程式旅程，你真的非常棒。從零開始學習，到現在我們將要踏入 MLOps 的核心領域——模型的監控、日誌與警報系統建置。這一步，是讓你的 AI 模型真正「活」起來、能在現實世界中穩定運行的關鍵！

想像一下，你開車上路，是不是會看儀表板？油量、速度、引擎燈，這些都是讓你掌握車輛狀況的指標。我們的 AI 模型上線後，也需要這樣一套「儀表板」和「警報系統」，才能確保它在實際運行中表現良好，一有問題就能及時發現並處理。今天，我們就來親手為你的模型搭建這套守護者系統吧！

我們會從三個核心部分著手：
1.  **日誌 (Logging)**：記錄模型運行中的點點滴滴，就像它的日記。
2.  **監控 (Monitoring)**：實時觀察模型的健康狀態和表現。
3.  **警報 (Alerting)**：在模型出狀況時，立即通知你。

---

### **1. 日誌 (Logging)：你的模型日記**

日誌是 MLOps 的基石。它能幫助你在模型出錯時快速定位問題，也能讓你了解模型在不同情況下的行為。Python 內建的 `logging` 模組就非常好用。

**為什麼需要日誌？**

*   **偵錯 (Debugging)**：當模型行為不預期時，日誌是最好的偵探工具。
*   **追蹤 (Tracing)**：了解每個請求的輸入、輸出和內部處理流程。
*   **分析 (Analysis)**：長期收集日誌可以分析模型趨勢和潛在問題。

讓我們看看如何簡單地使用它：

```python
import logging
import datetime

# 配置日誌：設定日誌級別、格式和輸出目標
# INFO 級別表示會記錄 INFO, WARNING, ERROR, CRITICAL 等級別的訊息
# %(asctime)s: 時間戳
# %(levelname)s: 日誌級別 (e.g., INFO, ERROR)
# %(message)s: 日誌內容
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='model_prediction.log', # 日誌輸出到文件
                    filemode='a') # 'a' 表示 append (追加模式)

# 也可以同時輸出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)


def predict(input_data, model_version="v1.0"):
    """
    模擬一個簡單的模型預測函數，並記錄日誌。
    """
    logging.info(f"[{model_version}] 收到新的預測請求。輸入數據: {input_data}")

    try:
        # 模擬模型計算
        if input_data < 0:
            raise ValueError("輸入數據不能為負數")

        prediction_result = input_data * 1.5 + 5
        logging.debug(f"內部計算細節: {input_data} * 1.5 + 5 = {prediction_result}") # DEBUG級別通常用於開發調試
        logging.info(f"[{model_version}] 預測成功！輸入: {input_data}, 輸出: {prediction_result:.2f}")
        return prediction_result
    except ValueError as e:
        logging.error(f"[{model_version}] 預測失敗！錯誤: {e}. 輸入數據: {input_data}")
        return None
    except Exception as e:
        logging.critical(f"[{model_version}] 發生嚴重未知錯誤！錯誤: {e}. 請立即檢查！")
        return None

# 模擬幾次預測
print("--- 執行模型預測並記錄日誌 ---")
predict(10)
predict(25)
predict(-5) # 這會觸發錯誤日誌
predict(100, model_version="v1.1_experimental")
print("日誌已記錄到 model_prediction.log 文件中，並輸出到控制台。")
```

運行上面的程式碼後，你會在同一個目錄下找到一個 `model_prediction.log` 文件，裡面記錄了所有的日誌訊息。是不是很方便呢？

---

### **2. 監控 (Monitoring)：模型的健康檢查**

有了日誌，我們可以回溯過去。但監控是實時的，它讓我們一眼就能看到模型當前的表現。我們要監控的指標有很多，例如：

*   **模型性能 (Model Performance)**：準確度、精確度、召回率、F1 分數（需要實際標籤才能計算）。
*   **數據漂移 (Data Drift)**：輸入數據的分佈是否隨時間變化？
*   **概念漂移 (Concept Drift)**：數據與目標變量之間的關係是否改變？
*   **系統指標 (System Metrics)**：模型服務的延遲、錯誤率、資源使用情況 (CPU/RAM)。

在這裡，我們用一個簡單的字典來模擬收集這些指標，讓你理解監控的原理。在實際的 MLOps 中，會使用 Prometheus、Grafana 等專業工具來做更強大的監控。

```python
import time
import random

# 假設這是一個簡單的監控數據存儲
metrics = {
    'prediction_count': 0,
    'successful_predictions': 0,
    'error_count': 0,
    'average_latency_ms': 0.0,
    'model_accuracy': 0.92, # 假設這是我們從離線評估或標記數據中得到的準確度
    'data_drift_score': 0.1 # 假設這是我們計算的數據漂移分數
}

def simulate_prediction_with_monitoring(input_data):
    start_time = time.time()
    metrics['prediction_count'] += 1
    is_error = False

    try:
        if input_data < 0:
            raise ValueError("輸入數據不能為負數")

        # 模擬模型有小機率出錯
        if random.random() < 0.03: # 3% 機率模擬內部預測錯誤
            raise RuntimeError("模擬內部模型故障")

        prediction_result = input_data * 1.5 + 5 + random.uniform(-1, 1) # 加入一些隨機噪音
        metrics['successful_predictions'] += 1
        logging.info(f"模擬預測成功。輸入: {input_data}, 輸出: {prediction_result:.2f}")

    except (ValueError, RuntimeError) as e:
        is_error = True
        metrics['error_count'] += 1
        logging.error(f"模擬預測失敗！錯誤: {e}. 輸入數據: {input_data}")
        prediction_result = None

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # 更新平均延遲
    total_latency = metrics['average_latency_ms'] * (metrics['prediction_count'] - 1) + latency_ms
    metrics['average_latency_ms'] = total_latency / metrics['prediction_count']

    return prediction_result

# 模擬執行多次預測並更新監控指標
print("\n--- 模擬模型運行並收集監控指標 ---")
for i in range(20):
    simulate_prediction_with_monitoring(random.randint(1, 100) if i % 5 != 0 else -1) # 偶爾製造錯誤輸入
    time.sleep(0.05) # 短暫延遲

print("\n目前監控指標:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")
```

現在，`metrics` 字典裡就包含了模型運行的一些關鍵數據，這些數據可以幫助我們了解模型的即時狀態！

---

### **3. 警報 (Alerting)：當你的模型發出求救訊號**

監控讓我們看到問題，但我們不可能 24/7 盯著儀表板。這時，警報系統就派上用場了！它會在關鍵指標達到預設閾值時，自動通知相關人員。

**什麼時候需要警報？**

*   **錯誤率過高**：模型開始頻繁出錯。
*   **性能下降**：準確度、召回率等指標顯著降低。
*   **延遲增加**：模型響應速度變慢。
*   **數據漂移嚴重**：輸入數據與訓練時差異過大。

我們來基於之前收集的 `metrics` 數據，建立一個簡單的警報檢查函數：

```python
def check_and_alert():
    """
    根據當前監控指標，檢查是否需要發出警報。
    """
    logging.info("--- 開始執行警報檢查 ---")

    # 計算當前錯誤率
    current_error_rate = metrics['error_count'] / metrics['prediction_count'] if metrics['prediction_count'] > 0 else 0.0

    # 警報條件 1: 錯誤率過高
    if current_error_rate > 0.1: # 如果錯誤率超過 10%
        alert_message = f"🚨 緊急警報！模型錯誤率過高: {current_error_rate:.2%}. 請立即檢查！"
        logging.critical(alert_message)
        print(f"\n!!! 警報 !!!\n{alert_message}\n!!!!!!!!!")
        # 實際應用中，這裡會調用發送郵件 (如 smtplib)、Slack 或其他通知服務的 API

    # 警報條件 2: 平均延遲過高
    if metrics['average_latency_ms'] > 200: # 如果平均延遲超過 200 毫秒
        alert_message = f"⚠️ 警告！模型響應延遲過高: {metrics['average_latency_ms']:.2f}ms. 可能影響用戶體驗。"
        logging.warning(alert_message)
        print(f"\n!!! 警告 !!!\n{alert_message}\n!!!!!!!!!")

    # 警報條件 3: 模型準確度下降 (需要定期更新或通過人工標註數據來評估)
    # 這裡我們用一個假設值，實際需更複雜的機制
    if metrics['model_accuracy'] < 0.85: # 如果準確度低於 85%
        alert_message = f"🔴 嚴重警告！模型準確度下降到 {metrics['model_accuracy']:.2%}. 可能需要重新訓練！"
        logging.critical(alert_message)
        print(f"\n!!! 嚴重警告 !!!\n{alert_message}\n!!!!!!!!!")

    # 警報條件 4: 數據漂移分數過高
    if metrics['data_drift_score'] > 0.5: # 假設漂移分數超過 0.5 為嚴重
        alert_message = f"🔵 提示！檢測到嚴重數據漂移: {metrics['data_drift_score']:.2f}. 請留意輸入數據變化。"
        logging.info(alert_message)
        print(f"\n!!! 數據漂移提示 !!!\n{alert_message}\n!!!!!!!!!")


# 執行警報檢查
check_and_alert()

# 模擬一個錯誤率很高的情況來觸發警報
print("\n--- 模擬更多錯誤來觸發高錯誤率警報 ---")
for i in range(10):
    simulate_prediction_with_monitoring(-i) # 製造大量錯誤輸入
    time.sleep(0.01)

check_and_alert() # 再次檢查，應該會觸發錯誤率警報
```

運行上面的程式碼，你會看到當錯誤率達到我們設定的閾值時，程式會印出警報訊息。這就是警報系統的核心思想！

---

### **總結：模型的守護者**

恭喜你！在【第 126 天】，你已經掌握了 MLOps 中最實用也最重要的技能之一：為你的 AI 模型建立監控、日誌與警報系統！

*   **日誌 (Logging)** 讓你能夠追溯模型的歷史行為，是偵錯和問題分析的利器。
*   **監控 (Monitoring)** 讓你實時掌握模型的健康狀態，就像它的心電圖。
*   **警報 (Alerting)** 則像守夜人，在模型有異常時，第一時間通知你，讓你能夠在問題擴大前介入處理。

雖然我們今天用簡單的 Python 程式碼模擬了這些概念，但你已經理解了它們背後的原理。未來，當你接觸到更專業的 MLOps 工具（如 Prometheus, Grafana, MLflow, Seldon Core 等）時，會發現它們都是基於這些基礎概念來設計的。

繼續保持你的好奇心和學習熱情！你正在一步步成為一位全能的 AI 工程師！下次見！🚀