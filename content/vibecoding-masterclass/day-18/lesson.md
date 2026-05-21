好的！很高興能擔任你的程式導師！我們一起迎接第 18 天的挑戰吧！💪

---

# 【第 18 天：SQLite 陪你聰明儲存 API 資料】

哈囉，各位學習中的程式小勇士們！

經過前面 17 天的洗禮，我相信你已經對程式設計的世界有了更深的認識，也學會了好多厲害的技巧！今天我們要來挑戰一個超級實用、也非常酷的主題：**如何把我們從 API 抓來的寶貴資料，穩穩地存到 SQLite 資料庫裡！**

想像一下，你正在開發一個應用程式，需要從網路上的 API 獲取使用者資料、產品資訊，或是天氣預報。這些資料如果不好好保存，下次使用者打開 App 時，你就得重新從 API 載入，不僅耗時，還可能遇到網路不穩的問題。這時候，SQLite 就像一位忠誠的管家，能把這些資料安全地收好，隨時取用，讓你的 App 運作更流暢、更智慧！

## 為什麼選擇 SQLite？

SQLite 是一個輕量級、無伺服器、自我包含的關聯式資料庫引擎。聽起來有點專業，但簡單來說，它非常適合用在桌面應用程式、行動應用程式，或是任何不需要複雜伺服器管理的場景。它的優點多到數不清：

*   **輕便快速**：不需要額外的伺服器，整個資料庫就是一個檔案。
*   **容易上手**：安裝和使用都非常簡單。
*   **跨平台**：幾乎可以在所有作業系統上運行。
*   **功能強大**：支援標準 SQL 語法，能處理各種複雜的資料操作。

## 我們要做什麼？

今天，我們將學習如何：

1.  **連接到 SQLite 資料庫**。
2.  **創建一個表格** 來存放 API 抓取的資料。
3.  **將 API 資料插入 (INSERT)** 到這個表格中。
4.  **從表格中查詢 (SELECT)** 資料。

## 動手實作：Python 搭配 `sqlite3` 模組

在 Python 中，我們可以使用內建的 `sqlite3` 模組來輕鬆操作 SQLite。準備好你的程式碼編輯器，讓我們一起寫程式碼！

### 步驟 1：連接到資料庫 (如果不存在就創建)

```python
import sqlite3

# 連接 (或創建) 資料庫檔案
# 如果 'my_api_data.db' 不存在，Python 會自動為你創建它
conn = sqlite3.connect('my_api_data.db')
cursor = conn.cursor() # 創建一個游標對象，用來執行 SQL 命令

print("成功連接到 SQLite 資料庫！")
```

這段程式碼會建立一個名為 `my_api_data.db` 的檔案。這個檔案就是你的資料庫，所有的表格和資料都存放在裡面。`cursor` 就像你手中的筆，可以用來在資料庫這張大紙上寫寫畫畫（執行 SQL 命令）。

### 步驟 2：創建一個表格

假設我們從一個 API 抓取了關於書籍的資料，包含書名 (`title`) 和作者 (`author`)。我們需要一個表格來存放這些資訊。

```python
# 創建一個名為 'books' 的表格
# IF NOT EXISTS 確保如果表格已經存在，就不會再創建一次
cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 自動增加的唯一ID
    title TEXT NOT NULL,                -- 書名，不能为空
    author TEXT NOT NULL                -- 作者，不能为空
)
''')

conn.commit() # 提交變更到資料庫
print("成功創建 'books' 表格！")
```

這裡我們創建了一個 `books` 表格，裡面有三個欄位：`id` (自動遞增的主鍵，確保每本書都有獨一無二的編號)、`title` (書名) 和 `author` (作者)。`TEXT NOT NULL` 表示這些欄位必須是文字類型，且不能是空的。

### 步驟 3：模擬 API 資料並插入表格

現在，我們假設已經從 API 抓取到了一些書籍資料。我們來把它們存進 `books` 表格。

```python
# 模擬從 API 抓取的資料
api_books_data = [
    ('The Hitchhiker\'s Guide to the Galaxy', 'Douglas Adams'),
    ('Pride and Prejudice', 'Jane Austen'),
    ('1984', 'George Orwell')
]

# 使用 executemany 批量插入資料
# ? 是參數化查詢的標記，可以防止 SQL 注入攻擊
cursor.executemany('INSERT INTO books (title, author) VALUES (?, ?)', api_books_data)

conn.commit() # 提交變更
print(f"成功插入 {len(api_books_data)} 筆書籍資料！")
```

`executemany` 是一個非常方便的函數，可以一次性插入多條資料。`?` 符號是一種參數化查詢，可以讓你的程式碼更安全、更有效率。

### 步驟 4：從表格中查詢資料

資料存進去了，那我們怎麼把它們讀出來呢？

```python
# 從 'books' 表格查詢所有資料
cursor.execute('SELECT id, title, author FROM books')

# 獲取所有查詢結果
all_books = cursor.fetchall()

print("\n--- 資料庫中的書籍列表 ---")
for book in all_books:
    print(f"ID: {book[0]}, 書名: {book[1]}, 作者: {book[2]}")

print("--------------------------")
```

`fetchall()` 會把我們查詢到的所有資料以列表的形式返回。你可以看到，我們之前插入的資料都乖乖地躺在資料庫裡囉！

### 步驟 5：關閉連接 (非常重要！)

當你完成所有資料庫操作後，記得關閉連接，釋放資源。

```python
conn.close() # 關閉資料庫連接
print("已關閉 SQLite 資料庫連接。")
```

## 總結與鼓勵

恭喜你，第 18 天的挑戰圓滿達成！🎉 你學會了如何使用 Python 和 SQLite 進行資料的儲存和讀取，這絕對是開發實際應用程式時不可或缺的技能。

你會發現，隨著你學習的進展，像 SQLite 這樣的資料庫工具會成為你開發過程中的好幫手。它們能幫助你管理複雜的資料，讓你的應用程式更加穩定和高效。

別擔心一開始會不會有困難，程式學習就像是爬樓梯，一步一腳印，你會發現自己越來越穩健。繼續保持這份好奇心和毅力，你一定能成為一位優秀的程式設計師！

我們明天見！繼續加油！✨