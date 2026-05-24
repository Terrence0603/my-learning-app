恭喜你！來到程式學習的第 21 天，這是一個值得慶祝的里程碑！你已經累積了不少程式基礎，現在是時候讓你的技能更上一層樓，面對真實世界的資料挑戰了。

今天我們要來學習一項超級實用的技能：如何使用 **Pandas** 這個強大的 Python 函式庫，來整理與分析從 **API** 獲取的資料。API 就像是應用程式之間的橋樑，能讓我們輕鬆取得各種網路上的資訊。而 Pandas 呢，則是處理這些資料的瑞士刀，能讓混亂的資料變得井然有序，並從中找出有價值的洞察。

別擔心，這一切聽起來可能有點複雜，但跟著我的步驟，你會發現它其實非常有趣，而且你將會掌握一項超能力！

---

## 【第 21 天：使用 Pandas 整理與分析 API 資料 - 讓資料跳舞吧！】

### 1. 為什麼需要 Pandas？

想像一下，你透過 API 拿到了一大堆資料，它們通常是 `JSON` 格式，看起來像一堆嵌套的字典和列表。原始的 JSON 資料雖然有用，但對於視覺化分析或複雜的篩選操作來說，就不那麼友善了。

這時候，Pandas 就會是你的救星！它能將這些資料轉換成類似試算表（Excel）的表格形式，也就是我們常說的 **DataFrame**。一旦資料進入 DataFrame，你就可以輕鬆地進行篩選、排序、計算、分組等各種操作，就像是在玩遊戲一樣！

### 2. 環境準備：安裝必要的函式庫

在開始之前，請確保你的 Python 環境中已經安裝了 `requests` 和 `pandas` 這兩個函式庫。如果沒有，請打開你的終端機或命令提示字元，輸入以下指令：

```bash
pip install requests pandas
```

### 3. 第一步：從 API 取得資料

我們將使用一個簡單的免費 API 服務：[JSONPlaceholder](https://jsonplaceholder.typicode.com/)。它提供了一些假資料供開發者測試。今天，我們將取得他們的 `/todos` (待辦事項) 資料。

```python
import requests
import pandas as pd

# 1. 定義 API 網址
api_url = "https://jsonplaceholder.typicode.com/todos"

# 2. 發送 HTTP GET 請求
response = requests.get(api_url)

# 3. 檢查請求是否成功 (狀態碼 200 表示成功)
if response.status_code == 200:
    # 4. 將回應內容解析成 Python 的字典或列表
    todos_data = response.json()
    print("成功取得資料！資料類型：", type(todos_data))
    print("前兩筆資料範例：")
    print(todos_data[:2]) # 顯示前兩筆資料
else:
    print(f"取得資料失敗！狀態碼：{response.status_code}")
    todos_data = [] # 如果失敗，設定為空列表
```

執行上面的程式碼，你會看到 `todos_data` 是一個包含多個字典的列表，每個字典代表一個待辦事項。看起來還算整齊，但如果資料量龐大，或者需要進行複雜分析時，它就不那麼方便了。

### 4. 第二步：將 JSON 資料轉換為 Pandas DataFrame

現在，就是 Pandas 展現魔法的時候了！我們將 `todos_data` 這個列表轉換成 DataFrame：

```python
# 5. 將資料轉換成 Pandas DataFrame
df = pd.DataFrame(todos_data)

print("\n轉換成 DataFrame 後的資料：")
# 6. 使用 .head() 方法快速預覽前幾行資料
print(df.head())

# 7. 查看 DataFrame 的概況 (包含欄位名稱、非空值數量、資料類型等)
print("\nDataFrame 資訊概覽：")
df.info()
```

是不是很神奇？原本的字典列表，現在變成了一個整齊的表格！`df.head()` 讓你快速瀏覽資料的開頭，而 `df.info()` 則提供了資料的概況，例如有多少筆資料、每個欄位的資料類型是什麼等等。

從 `df.head()` 的輸出，我們可以看到幾個欄位：`userId` (使用者 ID)、`id` (待辦事項 ID)、`title` (標題) 和 `completed` (是否已完成，布林值)。

### 5. 第三步：探索與分析資料

現在資料在 DataFrame 中，我們可以開始玩轉它了！

#### 範例一：統計已完成/未完成的待辦事項

我們可以使用 `value_counts()` 方法來統計 `completed` 欄位中 `True` (已完成) 和 `False` (未完成) 的數量。

```python
print("\n--- 待辦事項完成狀態統計 ---")
completion_counts = df['completed'].value_counts()
print(completion_counts)
```

從結果你會看到，`True` 和 `False` 的數量各佔一半，這很可能是這個假資料服務的設定。

#### 範例二：找出所有已完成的待辦事項

如果你只想看那些已經完成的待辦事項，篩選非常簡單：

```python
print("\n--- 所有已完成的待辦事項 ---")
completed_tasks = df[df['completed'] == True]
print(completed_tasks.head()) # 只顯示前五筆已完成的
print(f"總共有 {len(completed_tasks)} 筆已完成的待辦事項。")
```

這裡 `df[df['completed'] == True]` 的意思是：從 `df` 中，選出所有 `completed` 欄位為 `True` 的那些行。

#### 範例三：計算每個使用者的待辦事項完成率

這是 Pandas 強大的分組 (groupby) 功能的一個應用。我們想知道每個 `userId` 完成了多少比例的待辦事項。

```python
print("\n--- 每個使用者的待辦事項完成率 ---")
# 將 'completed' 欄位轉為數值 (True=1, False=0)，然後計算平均值
# 平均值即為完成率
user_completion_rate = df.groupby('userId')['completed'].mean()
print(user_completion_rate)
```

`df.groupby('userId')` 會根據 `userId` 將資料分成不同的組。然後 `['completed'].mean()` 會對每個組內的 `completed` 欄位計算平均值。因為 `True` 會被當作 `1`，`False` 當作 `0`，所以平均值就是完成率！超讚的！

---

### 總結

恭喜你！在今天的課程中，你已經掌握了一項超能力：
1.  **從 API 獲取 JSON 資料。**
2.  **使用 Pandas 將 JSON 轉換成易於操作的 DataFrame。**
3.  **對 DataFrame 進行基本的探索、篩選和分組分析。**

這只是 Pandas 冰山一角的功能，但你已經走出了最關鍵的第一步。從現在開始，當你遇到 API 資料時，你不再需要手動整理，而是可以讓 Pandas 成為你的得力助手，讓資料為你說話。

**小練習：**
*   嘗試修改 API 網址，看看 JSONPlaceholder 的 `/posts` 或 `/users` 會有什麼資料？
*   練習使用 Pandas 找出完成率最高的 `userId` 是誰。
*   嘗試篩選出 `userId` 為 `1` 且 `completed` 為 `False` 的待辦事項。

堅持下去，你將會成為一名資料整理與分析的大師！期待你在第 22 天的精彩表現！