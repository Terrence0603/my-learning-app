哈囉，我的程式學習夥伴！👋 歡迎來到我們的「Python 學習之旅」第 22 天！

你是不是覺得時間過得飛快？不知不覺間，我們已經從最基本的變數、迴圈，一路學到如何跟遠端的 API 溝通，取得寶貴的網路資料了！你真的很棒！🎉

今天，我們要來讓這些從 API 辛苦取得的資料「活起來」！想像一下，如果我們只是看到一堆密密麻麻的文字或數字，是不是很難一眼看出其中的規律或趨勢？這時候，「視覺化」就像是魔法一樣，能讓資料開口說話，把複雜的資訊變成一眼就能理解的圖表！

---

## 🚀 第 22 天：用圖表讓 API 資料開口說話！

### 🎯 今日目標：
1.  理解資料視覺化的重要性。
2.  學習使用 `matplotlib` 函式庫來繪製基本圖表。
3.  將我們從 API 取得的資料，透過圖表呈現分析結果。

---

### 為什麼需要視覺化？

想想看，如果我給你一份包含 1000 筆商品銷售記錄的試算表，和一張展示銷售額增長趨勢的折線圖，你覺得哪一個能讓你更快地理解「這家公司最近銷售狀況很好」？答案顯而易見是圖表！

視覺化能夠：
*   **快速洞察**：一眼看出資料中的趨勢、模式、異常點。
*   **溝通效率**：圖表比文字和數字更容易理解，也更容易向他人解釋你的發現。
*   **發現問題**：隱藏在數字中的問題，往往會在圖表中顯現出來。

在 Python 中，`matplotlib` 是最經典也最基礎的視覺化函式庫，而 `seaborn` 則是建立在 `matplotlib` 之上，提供更美觀、統計圖表更豐富的選擇。今天，我們就先從 `matplotlib` 入門！

---

### 🛠️ 事前準備：安裝必要的函式庫

在開始之前，請確保你已經安裝了以下函式庫：
*   `requests`：用於發送 API 請求。
*   `pandas`：強大的資料處理函式庫，能讓我們的資料分析更方便。
*   `matplotlib`：我們的視覺化主力！

如果你還沒安裝，可以在你的終端機或命令提示字元中執行：

```bash
pip install requests pandas matplotlib
```

---

### 實作範例：分析 JSONPlaceholder 的貼文資料

我們將繼續使用之前介紹過的免費 API 服務：[JSONPlaceholder](https://jsonplaceholder.typicode.com/)。這次，我們來取得所有的「貼文 (posts)」資料，然後分析「每個使用者發布了多少篇貼文」，並用長條圖呈現出來！

#### 步驟 1：取得 API 資料

首先，像我們之前學過的一樣，用 `requests` 函式庫去取得貼文資料。

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

# JSONPlaceholder 的貼文 API 網址
api_url = "https://jsonplaceholder.typicode.com/posts"

try:
    response = requests.get(api_url)
    response.raise_for_status() # 檢查請求是否成功，如果失敗會拋出 HTTPError
    posts_data = response.json()

    print(f"✅ 成功取得 {len(posts_data)} 筆貼文資料。")
    # 偷偷看一下第一筆資料長什麼樣子
    # print("\n第一筆貼文資料範例：")
    # print(posts_data[0])

except requests.exceptions.RequestException as e:
    print(f"❌ 請求 API 失敗：{e}")
    posts_data = [] # 確保 posts_data 是一個空列表，避免後續程式碼出錯
```

你會看到類似這樣的資料結構（每筆貼文都包含 `userId` 和 `id` 等資訊）：
```json
{
  "userId": 1,
  "id": 1,
  "title": "sunt aut facere ...",
  "body": "quia et suscipit ..."
}
```

#### 步驟 2：進行資料分析（計算每個使用者的貼文數量）

現在我們有了所有的貼文資料，接下來就是分析的環節。我們要統計每個 `userId` 出現的次數。使用 Pandas 的 DataFrame 會讓這個過程變得非常簡單！

```python
if posts_data: # 確保有資料才進行分析
    # 將資料轉換為 Pandas DataFrame，方便處理
    df = pd.DataFrame(posts_data)

    # 計算每個使用者發布的貼文數量
    # value_counts() 會計算 Series 中每個唯一值的出現次數
    # sort_index() 則會依據使用者 ID 排序，讓圖表更整齊
    posts_per_user = df['userId'].value_counts().sort_index()

    print("\n📊 每個使用者發布的貼文數量：")
    print(posts_per_user)
else:
    print("沒有資料可供分析。")
```

執行後，你會看到類似這樣的結果：
```
📊 每個使用者發布的貼文數量：
userId
1     10
2     10
3     10
4     10
5     10
6     10
7     10
8     10
9     10
10    10
Name: count, dtype: int64
```
看起來 JSONPlaceholder 的資料很平衡，每個使用者都發了 10 篇貼文！這很適合用來做長條圖。

#### 步驟 3：使用 `matplotlib` 繪製長條圖

重頭戲來了！我們要用 `matplotlib` 把這個分析結果畫出來。長條圖 (Bar Chart) 是呈現類別資料數量分佈的絕佳選擇。

```python
if not posts_per_user.empty: # 確保分析結果不為空
    # 設定圖表大小，讓它看起來更舒服
    plt.figure(figsize=(10, 6))

    # 繪製長條圖
    # x 軸是使用者 ID (posts_per_user.index)，需要轉換成字串，以免 matplotlib 誤解為數值範圍
    # y 軸是貼文數量 (posts_per_user.values)
    plt.bar(posts_per_user.index.astype(str), posts_per_user.values, color='skyblue')

    # 添加圖表標題
    plt.title('使用者發文數量分佈', fontsize=16)

    # 添加 x 軸和 y 軸的標籤
    plt.xlabel('使用者 ID', fontsize=12)
    plt.ylabel('貼文數量', fontsize=12)

    # 調整 x 軸刻度，rotation=45 讓標籤傾斜，ha='right' 讓標籤右對齊，避免重疊
    plt.xticks(rotation=45, ha='right')

    # 顯示 y 軸的網格線，幫助閱讀數值
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 自動調整圖表邊距，確保所有元素都顯示完整
    plt.tight_layout()

    # 顯示圖表！這一步是真正把圖表顯示出來的魔法
    plt.show()
else:
    print("沒有使用者貼文資料可供繪製圖表。")
```

執行這段程式碼後，你的螢幕上應該會彈出一個視窗，顯示著一張清晰的長條圖，是不是很有成就感！

---

### 🎨 更多探索與挑戰

*   **嘗試不同圖表類型**：`matplotlib` 不只有長條圖，還有折線圖 (`plt.plot()`)、散佈圖 (`plt.scatter()`)、圓餅圖 (`plt.pie()`) 等等。你可以嘗試分析不同的 API 資料，並選擇最適合的圖表類型。
*   **美化你的圖表**：`matplotlib` 提供了大量的參數來自定義顏色、字體、邊框、圖例等等。你可以查閱官方文件，讓你的圖表更具專業感。
*   **進階視覺化**：當你對 `matplotlib` 有了一定了解後，可以嘗試 `seaborn` 函式庫。它能用更簡潔的程式碼繪製出更統計學導向且美觀的圖表。
*   **分析更多資料**：JSONPlaceholder 還有 `todos` (待辦事項)、`users` (使用者資訊) 等 API。你可以試著取得這些資料，然後分析「每個使用者完成多少待辦事項」、「不同使用者名稱的長度分佈」等等。

---

### 總結與鼓勵

恭喜你！今天你學會了如何讓冰冷的 API 資料變成生動的圖表，這就像是給你的資料加上了眼睛和聲音，讓它們能夠「說故事」！從今天開始，你不再只是資料的接收者，更是資料的詮釋者和傳播者。

視覺化是資料分析中不可或缺的一環，它能大大提升你理解和溝通資料的能力。這是一項非常實用且強大的技能，你現在已經掌握了它的基礎！

繼續保持這份好奇心和探索精神吧！資料的世界充滿了樂趣，你會發現更多的魔法！

明天見！🚀