哈囉，未來的大大們！👋 歡迎來到我們的程式學習之旅第 16 天！

你是不是常常聽到「API」這個詞，感覺它既神祕又強大？沒錯，API (Application Programming Interface) 就像是不同應用程式之間的橋樑，讓它們可以互相溝通、交換資料。想像一下，你不需要知道餐廳廚房怎麼煮菜，只要跟服務生（API）點餐，就能享用美食。

而今天，我們要學習如何使用 Python 界最受歡迎、也最直覺的 HTTP 函式庫 —— **Requests**，來扮演這位「服務生」，跟網路上的各種服務（API）點餐（發送請求）！準備好了嗎？讓我們一起打開與世界溝通的大門吧！

---

## 【第 16 天：使用 Requests 函式庫串接 API】

### 一、Requests 函式庫，你的 API 魔法棒 ✨

在 Python 中，Requests 函式庫讓發送 HTTP 請求變得非常簡單。不管是取得資料、上傳資料，它都能輕鬆搞定。

#### 1. 安裝 Requests

首先，我們需要把 Requests 函式庫安裝到你的環境中。打開你的終端機或命令提示字元，輸入：

```bash
pip install requests
```

搞定！是不是超級簡單？

### 二、發出你的第一個 GET 請求：跟 API 說「我要！」

最常見的 API 請求類型就是 **GET**，它就像是去網路上的圖書館，向館員（API）索取書籍（資料）。我們今天要使用的圖書館是一個很棒的測試用 API：[JSONPlaceholder](https://jsonplaceholder.typicode.com/)。它提供了一些假的貼文、使用者等資料，非常適合我們練習。

讓我們試著取得一篇貼文的資料：

```python
import requests # 導入 requests 函式庫

# 我們的目標 API 網址 (Endpoint)
# 這個網址會回傳 ID 為 1 的貼文資料
url = "https://jsonplaceholder.typicode.com/posts/1"

# 發出 GET 請求
print(f"正在向 {url} 發出 GET 請求...")
response = requests.get(url)

# 檢查回應狀態碼 (Status Code)
# 200 表示成功，404 表示找不到，500 表示伺服器錯誤等等
print(f"狀態碼：{response.status_code}")

# 顯示回應內容 (文字形式)
print("\n--- 原始回應內容 ---")
print(response.text)

# 如果狀態碼是 200，表示成功取得資料
if response.status_code == 200:
    print("\n恭喜你，成功取得資料囉！")
else:
    print("\n請求失敗，請檢查狀態碼或網路連線。")
```

執行這段程式碼，你會看到：

*   狀態碼是 `200`，表示請求成功！
*   `response.text` 顯示了一段 JSON 格式的文字，這就是 API 回傳給我們的貼文資料。

太棒了！你已經成功發出了第一個 API 請求！

### 三、GET 請求帶參數：告訴 API 「我要那種！」

有時候，我們不只是想隨機拿資料，而是想指定條件，例如「給我 ID 是 1 的使用者發的所有貼文」。這時候就需要使用 **參數 (Parameters)**。

Requests 函式庫提供了一個很方便的方式來傳遞參數：

```python
import requests

# 我們的目標 API 網址 (這次沒有指定貼文 ID 了)
url = "https://jsonplaceholder.typicode.com/posts"

# 建立一個字典來存放我們的參數
# 我們想取得 userId 是 1 的所有文章
params = {
    "userId": 1
}

print(f"正在向 {url} 發出 GET 請求，參數為 {params}...")
response = requests.get(url, params=params) # 將 params 字典傳給 requests.get()

print(f"狀態碼：{response.status_code}")

# 這次的回應可能比較長，我們只顯示前 200 個字元
print("\n--- 原始回應內容 (部分) ---")
print(response.text[:200] + "...") # 避免印出太長的回應

if response.status_code == 200:
    print("\n成功取得特定使用者文章！")
else:
    print("\n請求失敗。")
```

看到了嗎？Requests 會自動幫我們把 `params` 字典轉換成 URL 後面的查詢字串（例如 `?userId=1`），讓你的程式碼更整潔、更易讀！

### 四、解析 JSON 資料：把資料變成 Python 能懂的字典！

API 回傳的資料通常是 JSON (JavaScript Object Notation) 格式。它看起來像 Python 的字典和列表，但本質上是字串。為了方便我們在 Python 中操作這些資料，Requests 提供了一個超好用的方法：`response.json()`。

```python
import requests

url = "https://jsonplaceholder.typicode.com/posts/1"
response = requests.get(url)

if response.status_code == 200:
    # 使用 .json() 方法將 JSON 回應轉換成 Python 字典或列表
    data = response.json()

    print("\n--- 解析後的 JSON 資料 ---")
    print(f"資料型態：{type(data)}") # 看一下轉換後是什麼型態 (應該是 dict)
    print(f"文章標題：{data['title']}")
    print(f"文章內文：{data['body'][:50]}...") # 取前50字顯示
    print(f"使用者ID：{data['userId']}")

    # 如果想看看所有的 key 和 value
    print("\n--- 所有資料欄位 ---")
    for key, value in data.items():
        print(f"{key}: {value}")

else:
    print("請求失敗，無法解析 JSON 資料。")
```

現在，`data` 變數就是一個標準的 Python 字典了！你可以像操作普通字典一樣，透過鍵 (key) 來取得對應的值 (value)，例如 `data['title']`。這讓處理 API 資料變得輕而易舉！

---

### 結語與挑戰！

恭喜你！今天我們學會了：

*   如何安裝 Requests 函式庫。
*   如何發送最基本的 GET 請求。
*   如何使用參數來篩選或指定資料。
*   最重要的是，如何將 API 回傳的 JSON 資料解析成 Python 字典，方便我們使用！

這只是 Requests 函式庫和 API 世界的冰山一角。還有 POST、PUT、DELETE 這些更進階的請求（就像是新增、修改、刪除資料），以及身份驗證、錯誤處理等等。

**小小挑戰：**

試著使用 `https://jsonplaceholder.typicode.com/users/1` 這個網址，取得 ID 為 1 的使用者資料，並印出他的姓名 (name)、電子郵件 (email) 和公司名稱 (company.name)！

繼續保持這份好奇心和學習熱情，一步一腳印，你一定能成為超級程式設計師！我們明天見！🚀