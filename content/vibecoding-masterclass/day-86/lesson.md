好的，我的未來 MLOps 大師！準備好迎接你的模型部署後，如何讓它們保持最佳狀態的秘密武器了嗎？

---

# 第 86 天：實戰：MLOps 模型監控與可觀測性 - 你的模型也能有「眼睛」！

哈囉，各位程式探險家！恭喜你走到第 86 天，這代表你已經對 AI 和機器學習的世界有了深入的理解。你可能已經學會了如何訓練出一個超棒的模型，把它部署上線。但，接下來呢？你的模型部署後，就永遠完美無缺、精準無比嗎？

答案是：**不！** （除非你有神力加持，哈哈）。

就像人類會感冒、汽車需要保養一樣，部署後的機器學習模型也需要定期的「健康檢查」和「眼睛」來觀察周遭的變化。這就是我們今天的主題：**模型監控 (Model Monitoring)** 與 **可觀測性 (Observability)** 在 MLOps 中的重要角色。

## 為什麼需要模型監控？

想像一下，你訓練了一個很棒的推薦系統，部署後一開始效果奇佳。但過了一段時間，用戶開始抱怨推薦不準了，甚至出現奇怪的結果。為什麼會這樣？

有幾個常見的「模型殺手」：

1.  **資料漂移 (Data Drift)**：最常見的狀況！生產環境中的輸入資料分佈，可能隨著時間、季節、趨勢、使用者行為等因素而改變，與你模型訓練時的資料分佈不再一致。
    *   **範例**：你用前一年的消費數據訓練了一個模型，但今年有新的促銷活動或疫情，消費模式大變。
2.  **概念漂移 (Concept Drift)**：輸入資料的意義雖然沒變，但真實世界中「目標」的定義或其與輸入的關係改變了。
    *   **範例**：垃圾郵件的定義可能改變，以前是垃圾郵件的內容，現在不見得是。
3.  **模型性能衰退 (Model Performance Degradation)**：當資料漂移或概念漂移發生時，模型的預測準確度、精確度、召回率等指標自然會下降。
4.  **系統問題 (System Issues)**：模型伺服器可能過載、記憶體洩漏、延遲增加等。雖然這更偏向傳統的監控，但在 MLOps 中，它也影響模型服務的可用性。

所以，模型監控就是為了解決這些問題，讓你的模型像長了「眼睛」一樣，能夠隨時感知這些變化，並及時提醒你採取行動！

## 我們的目標：模擬一個簡單的模型監控系統

在今天的實戰中，我們不會建立一個完整的 MLOps 監控平台 (那會是超複雜的專案！)，但我們會用 Python 模擬一個簡單的監控流程：

1.  模擬一個訓練好的模型。
2.  模擬模型在生產環境中接收資料並做出預測。
3.  記錄每一次預測的輸入資料和預測結果。
4.  實作一個簡單的監控功能，偵測輸入資料的統計量變化，以及模型性能的變化。

準備好了嗎？讓我們開始動手吧！

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime # 用於模擬時間戳

print("🎉 程式導師說：MLOps 監控之路，從這裡開始！")

# --- 步驟 1: 模擬一個訓練好的模型 ---
# 為了簡單起見，我們用一個簡單的邏輯迴歸模型和模擬數據。
# 想像這就是你辛苦訓練出來並部署的模型！

np.random.seed(42) # 設定隨機種子，讓結果可重現

# 模擬訓練數據：兩個特徵 (feature_0, feature_1)，一個二元目標 (label)
X_train = np.random.rand(100, 2) * 10 # 數據範圍 0-10
y_train = (X_train[:, 0] + X_train[:, 1] > 10).astype(int) # 簡單的分類邏輯

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"\n✅ 模型已訓練完畢！訓練資料 X_train 範例:\n{X_train[:3]}")
print(f"模型的權重: {model.coef_}, 截距: {model.intercept_}")

# --- 步驟 2 & 3: 模擬生產環境中的模型預測與日誌記錄 ---
# 我們將使用一個列表來儲存每次預測的日誌。
production_logs = []

def predict_and_log(data_point, true_label=None):
    """
    模擬模型接收新的數據點並進行預測，同時記錄相關資訊。
    data_point: 字典，包含 'features' 列表
    true_label: 可選，如果後續知道真實標籤，可以補上
    """
    features = np.array(data_point['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    log_entry = {
        'timestamp': datetime.datetime.now(), # 記錄預測時間
        'features': data_point['features'],   # 記錄輸入特徵
        'prediction': prediction,             # 記錄模型預測結果
        'probability': probability,           # 記錄預測機率
        'true_label': true_label              # 記錄真實標籤 (如果有的話)
    }
    production_logs.append(log_entry)
    # print(f"處理數據: {data_point['features']}, 預測: {prediction}")
    return prediction

# --- 步驟 4: 實作監控功能 ---

def monitor_predictions(baseline_logs, current_logs):
    """
    監控模型的輸入資料漂移、預測結果漂移和模型性能。
    baseline_logs: 基準期的日誌 (例如，部署初期表現良好的日誌)
    current_logs: 當前生產環境的日誌
    """
    if not baseline_logs or not current_logs:
        print("⚠️ 導師提醒：資料不足，無法進行監控。請累積更多日誌！")
        return

    baseline_df = pd.DataFrame(baseline_logs)
    current_df = pd.DataFrame(current_logs)

    # 從 'features' 列表提取單個特徵，例如 'feature_0'
    baseline_df['feature_0'] = baseline_df['features'].apply(lambda x: x[0])
    current_df['feature_0'] = current_df['features'].apply(lambda x: x[0])

    print("\n--- 📊 模型監控報告 ---")

    # 1. 監控輸入資料分佈 (以 feature_0 的平均值為例)
    baseline_feature_0_mean = baseline_df['feature_0'].mean()
    current_feature_0_mean = current_df['feature_0'].mean()

    print(f"\n➡️ 監控項目：輸入資料漂移 (Feature_0 平均值)")
    print(f"  基準期 'feature_0' 平均值: {baseline_feature_0_mean:.2f}")
    print(f"  當前生產期 'feature_0' 平均值: {current_feature_0_mean:.2f}")

    # 設定一個閾值來判斷是否有顯著漂移 (這個閾值通常需要根據業務和數據特性來設定)
    if abs(current_feature_0_mean - baseline_feature_0_mean) > 1.5: # 如果平均值變化超過 1.5
        print("🚨 警告：輸入資料 'feature_0' 可能發生了顯著漂移！請檢查上游數據來源。")
    else:
        print("  輸入資料 'feature_0' 狀況良好。")

    # 2. 監控模型預測結果分佈 (以預測值的平均值為例)
    baseline_pred_mean = baseline_df['prediction'].mean()
    current_pred_mean = current_df['prediction'].mean()

    print(f"\n➡️ 監控項目：模型預測漂移 (預測值平均值)")
    print(f"  基準期預測平均值: {baseline_pred_mean:.2f}")
    print(f"  當前生產期預測平均值: {current_pred_mean:.2f}")

    if abs(current_pred_mean - baseline_pred_mean) > 0.3: # 如果預測平均值變化超過 0.3
        print("🚨 警告：模型預測結果分佈可能發生了漂移！這可能是資料漂移或模型性能下降的徵兆。")
    else:
        print("  模型預測結果分佈穩定。")

    # 3. 監控模型性能 (需要真實標籤)
    # 我們假設真實標籤在預測後一段時間才會取得，所以只對有真實標籤的數據進行評估。
    current_with_labels = current_df.dropna(subset=['true_label'])
    if not current_with_labels.empty:
        accuracy = accuracy_score(current_with_labels['true_label'], current_with_labels['prediction'])
        print(f"\n➡️ 監控項目：模型準確度")
        print(f"  當前模型準確度: {accuracy:.2f}")
        if accuracy < 0.75: # 如果準確度低於 75%
            print("🚨 警告：模型準確度已低於可接受閾值！建議重新訓練或檢查原因。")
        else:
            print("  模型準確度表現良好。")
    else:
        print("\nℹ️ 提示：尚無足夠的真實標籤來評估模型性能。繼續累積中...")

# --- 模擬生產環境運作 ---

# 模擬初始部署後一段時間 (基準期) 的數據，這段時間模型表現良好。
print("\n--- 模擬基準期資料收集 (模型剛上線，表現良好) ---")
baseline_data_points = []
for _ in range(30):
    features = np.random.rand(2) * 10
    true_label = (features[0] + features[1] > 10).astype(int) # 假設這段時間的真實標籤
    predict_and_log({'features': features.tolist()}, true_label=true_label)
    baseline_data_points.append(production_logs[-1]) # 將日誌複製一份作為基準

# 模擬一段時間後，生產環境的數據發生了變化 (資料漂移)
print("\n--- 模擬當前生產期資料收集 (數據開始漂移...) ---")
current_data_points = []
for i in range(30):
    features = np.random.rand(2) * 10
    # 在這裡製造一個「資料漂移」：第一個特徵的值普遍變高了！
    if i > 10: # 假設在中間某個時間點開始漂移
        features[0] += 3 # 讓 feature_0 普遍增加 3
    true_label = (features[0] + features[1] > 10).astype(int) # 新的真實標籤
    predict_and_log({'features': features.tolist()}, true_label=true_label)
    current_data_points.append(production_logs[-1])

# --- 執行監控 ---
print("\n--- 執行模型監控 ---")
monitor_predictions(baseline_data_points, current_data_points)

print("\n--- 恭喜你！模型監控體驗完成！ ---")
print("🎉 導師鼓勵：你已經掌握了讓你的模型擁有「眼睛」和「耳朵」的基本方法！")
print("別忘了，持續學習和實踐是 MLOps 的核心精神！")
```

## 程式碼解說：

1.  **模擬模型與數據**：我們使用 `sklearn` 建立一個簡單的 `LogisticRegression` 模型，並用 `numpy` 隨機生成數據來訓練它。這代表你已經部署到生產環境的模型。
2.  **`predict_and_log` 函數**：這是核心！它模擬了模型收到新的輸入 `data_point`，做出 `prediction`，並將所有相關資訊 (時間戳、原始特徵、預測結果、預測機率、真實標籤) 存入 `production_logs` 列表。
3.  **`monitor_predictions` 函數**：
    *   它接收 `baseline_logs` (代表模型剛上線時表現良好的數據基準) 和 `current_logs` (當前生產環境的數據)。
    *   **輸入資料漂移監控**：我們提取了 `feature_0` 的平均值，比較基準期和當前期的差異。如果差異過大，就發出警告。
    *   **預測結果漂移監控**：類似地，我們比較了模型預測結果 (0 或 1) 的平均值。如果模型開始傾向於某一個類別，這也可能是問題。
    *   **模型性能監控**：這需要 `true_label` (真實標籤)。在生產環境中，真實標籤通常會有延遲，但一旦獲得，我們就可以計算模型在當前數據上的 `accuracy_score`。如果準確度低於預設閾值，就該警惕了！
4.  **模擬生產環境**：我們分兩個階段生成數據：
    *   **基準期**：正常數據，用於建立「正常」行為的標準。
    *   **當前生產期**：我們故意在數據中引入了「漂移」(`features[0] += 3`)，讓第一個特徵的平均值明顯變高，模擬真實世界中數據分佈的變化。

## 下一步：真實世界中的監控

今天的範例很簡單，但它展示了模型監控的核心思想。在實際的 MLOps 環境中，你會使用更強大、更專業的工具來實現這些功能：

*   **資料漂移偵測工具**：例如 `Evidently AI`、`whylogs` 等，它們能提供更豐富的統計指標和可視化來偵測數據漂移。
*   **性能監控與告警**：`MLflow Tracking`、`Prometheus` + `Grafana`、各種雲服務的監控工具 (AWS CloudWatch, Google Cloud Monitoring) 等，可以收集模型指標，建立儀表板，並在指標異常時發送告警。
*   **日誌系統**：更完善的日誌儲存與查詢系統 (如 `ELK Stack: Elasticsearch, Logstash, Kibana`)，方便你追蹤模型的行為和除錯。

## 導師的叮嚀

模型監控是 MLOps 中至關重要的一環。它將你的模型從「部署後就祈禱」的狀態，提升到「部署後持續觀察、迭代優化」的專業水準。有了這些「眼睛」，你就能及早發現問題，避免模型在生產環境中造成更大的損失。

保持好奇心，繼續探索 MLOps 的奇妙世界吧！你做得非常棒！