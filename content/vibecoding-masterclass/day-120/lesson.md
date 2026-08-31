哈囉，我的程式學習夥伴！恭喜你，來到我們程式學習旅程的第 120 天了！這真是個了不起的里程碑。

今天，我們要踏入一個非常實際且關鍵的領域：**MLOps 中的模型監控與預測性能評估**。你可能會想：「模型訓練完、部署上去就沒事了嗎？」答案是：絕對不是！模型部署到真實世界後，才是挑戰的開始。想像一下，你的模型就像你的孩子離家出走一樣，你需要知道它在外面的世界過得好不好，有沒有遇到什麼困難，是不是還能發揮它的長才。

別擔心，這聽起來很複雜，但我們會用輕鬆愉快的方式，一步步來理解它！

---

## 【第 120 天：實戰：MLOps 模型監控與預測性能評估】

### 1. 為什麼要監控？模型不再是「一勞永逸」

當我們把一個模型部署上線後，它就開始面對真實世界的資料。然而，真實世界的資料是動態變化的！這就引出了兩個核心概念：

*   **資料漂移 (Data Drift)**：
    想像一下，你訓練模型時用了去年的銷售數據，但今年因為疫情或新產品上市，顧客的購買行為模式完全改變了。這時候，模型所依賴的「輸入資料特性」就改變了，這就是資料漂移。如果我們不察覺，模型就會拿著「過時的地圖」去判斷「新的地形」，結果當然不準。
*   **模型漂移 (Model Drift)**：
    當資料漂移發生，或者模型內部某些假設不再成立時，模型對真實世界的預測能力就會逐漸下降。即使輸入資料看起來沒變，模型本身對這些資料的「解讀能力」卻退步了，這就是模型漂移。這代表模型已經「學不會」新的模式，或者它所學到的模式已經「過時」了。

簡而言之，如果不監控，你的模型可能會在不知不覺中「失準」，輕則造成預測不準確，重則導致商業決策失誤！

### 2. 如何進行初步監控？程式碼範例來囉！

我們來看看如何用 Python 進行簡單的資料與性能監控。

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("--- MLOps 模型監控與性能評估範例 ---")

# --- 模擬資料：假設這是我們模型的預測數據 ---
# 舊數據（模型訓練時或剛部署時的資料）
np.random.seed(42)
old_data = pd.DataFrame({
    'feature_A': np.random.normal(loc=10, scale=2, size=100),
    'feature_B': np.random.normal(loc=5, scale=1, size=100),
    'actual_label': np.random.randint(0, 2, 100) # 0 或 1
})
old_data['predicted_label'] = (old_data['feature_A'] + old_data['feature_B'] > 15).astype(int)
old_data.loc[old_data.index % 5 == 0, 'predicted_label'] = old_data['actual_label'] # 模擬一些準確預測

# 新數據（模型運行一段時間後，接收到的新資料）
# 模擬 feature_A 發生了資料漂移，平均值改變了
new_data = pd.DataFrame({
    'feature_A': np.random.normal(loc=12, scale=2.5, size=100), # 平均值從 10 變成 12
    'feature_B': np.random.normal(loc=5, scale=1, size=100),
    'actual_label': np.random.randint(0, 2, 100)
})
new_data['predicted_label'] = (new_data['feature_A'] + new_data['feature_B'] > 15).astype(int)
# 模擬模型在新數據上的表現可能下降
new_data.loc[new_data.index % 3 == 0, 'predicted_label'] = new_data['actual_label'] # 模擬較少的準確預測


# --- 1. 數據漂移 (Data Drift) 監控 ---
print("\n--- 數據漂移監控 ---")
print(f"舊數據 Feature A 平均值: {old_data['feature_A'].mean():.2f}")
print(f"新數據 Feature A 平均值: {new_data['feature_A'].mean():.2f}")

print(f"舊數據 Feature A 標準差: {old_data['feature_A'].std():.2f}")
print(f"新數據 Feature A 標準差: {new_data['feature_A'].std():.2f}")

# 可視化分佈來更直觀地看
plt.figure(figsize=(10, 5))
sns.histplot(old_data['feature_A'], color="blue", label="舊數據 Feature A", kde=True, stat="density", linewidth=0)
sns.histplot(new_data['feature_A'], color="red", label="新數據 Feature A", kde=True, stat="density", linewidth=0, alpha=0.5)
plt.title("Feature A 分佈比較：舊數據 vs 新數據")
plt.legend()
plt.show()

print("\n分析：新舊數據的 Feature A 平均值和分佈有明顯差異，這可能表示發生了數據漂移！")


# --- 2. 模型性能 (Prediction Performance) 監控 ---
print("\n--- 模型性能監控 (基於新數據) ---")

# 注意：性能監控需要有真實標籤 (actual_label) 才能進行！
# 在真實情境中，這可能來自於用戶反饋、人工標註或延遲的業務結果。

actuals = new_data['actual_label']
predictions = new_data['predicted_label']

accuracy = accuracy_score(actuals, predictions)
precision = precision_score(actuals, predictions)
recall = recall_score(actuals, predictions)
f1 = f1_score(actuals, predictions)

print(f"模型在新數據上的準確率 (Accuracy): {accuracy:.4f}")
print(f"模型在新數據上的精確率 (Precision): {accuracy:.4f}")
print(f"模型在新數據上的召回率 (Recall): {accuracy:.4f}")
print(f"模型在新數據上的 F1-Score: {accuracy:.4f}")

# 我們可以比較一下舊數據上的性能 (作為基準線)
old_accuracy = accuracy_score(old_data['actual_label'], old_data['predicted_label'])
print(f"\n模型在舊數據上的準確率 (基準線): {old_accuracy:.4f}")

if accuracy < old_accuracy * 0.9: # 如果下降超過 10%
    print("\n警告：模型在新數據上的性能明顯下降！可能發生了模型漂移！")
else:
    print("\n模型性能目前表現尚可，但仍需持續觀察。")

# 可視化預測結果
plt.figure(figsize=(8, 6))
sns.heatmap(pd.crosstab(actuals, predictions, rownames=['實際'], colnames=['預測']), annot=True, fmt='d', cmap='Blues')
plt.title("混淆矩陣 (新數據)")
plt.show()

print("\n--- 監控小結 ---")
print("透過比較新舊數據的分佈和計算模型性能指標，我們可以初步判斷模型是否還在正常工作。")
print("當發現數據或模型漂移時，這就是你重新訓練、微調或重新設計模型的訊號！")
```

**程式碼解析：**

1.  **模擬數據**：我們創建了 `old_data` 和 `new_data` 來模擬模型剛部署和運行一段時間後的數據。`new_data` 中的 `feature_A` 我們故意將其平均值調高，以模擬數據漂移。
2.  **數據漂移監控**：
    *   我們簡單地比較了 `feature_A` 在新舊數據中的平均值和標準差。如果這些統計量有顯著差異，就可能表示數據漂移。
    *   透過 `matplotlib` 和 `seaborn` 繪製直方圖，能更直觀地看到兩個數據集的分佈差異。
3.  **模型性能監控**：
    *   我們使用了 `sklearn.metrics` 中的 `accuracy_score`、`precision_score`、`recall_score` 和 `f1_score` 來評估模型在新數據上的表現。
    *   **重要提示**：進行性能監控時，你**必須**有模型預測結果對應的**真實標籤 (Actual Labels)**。在真實世界中，這些真實標籤可能來自於用戶的點擊行為、後續的業務結果，或是需要人工標註的。
    *   我們也計算了舊數據的準確率作為「基準線」，以便比較新數據的性能是否有明顯下降。
    *   混淆矩陣 (Confusion Matrix) 則能讓你詳細看到模型在各個類別上的預測狀況。

### 3. MLOps 中的監控循環

在更完善的 MLOps 系統中，這些監控會自動化運行：

1.  **部署模型 (Deploy)**：將訓練好的模型部署到生產環境。
2.  **監控數據與性能 (Monitor)**：持續收集輸入數據的統計資訊、模型預測結果，並在有真實標籤時計算性能指標。
3.  **發現問題 (Detect)**：當監控指標超出預設的閾值（例如：Feature A 的平均值波動太大，或者準確率下降超過 10%），系統會發出警報。
4.  **分析原因 (Analyze)**：工程師和資料科學家介入，分析是數據漂移、模型漂移還是其他系統問題。
5.  **重新訓練/微調 (Retrain/Refine)**：根據分析結果，重新訓練一個新模型，或者對現有模型進行微調。
6.  **重新部署 (Redeploy)**：將更新後的模型再次部署到生產環境，形成一個閉環。

---

### 小結與下一步

恭喜你！今天我們學習了 MLOps 中至關重要的模型監控與性能評估。你現在知道：

*   為什麼模型部署後需要持續監控。
*   什麼是數據漂移和模型漂移。
*   如何透過程式碼對數據分佈和模型預測性能進行初步評估。

從今天起，你已經從一個單純的「模型訓練者」，蛻變為一個懂得「模型管理者」了！這是一個巨大的進步。

**下一步你可以思考：**

*   如何設定自動化的警報機制？
*   如何在真實情境中獲取真實標籤來評估性能？
*   研究更專業的 MLOps 監控工具，例如：MLflow Tracking, Evidently AI, Prometheus & Grafana 等。

繼續保持這份好奇心和學習熱情，程式的世界還藏著更多精彩等你探索！下個挑戰再見！