哈囉，程式探險家們！ 👋

歡迎來到你的程式學習之旅【第 17 天】！

前幾天，我們一起學會了如何向遠方的伺服器「發出請求」，就像是點了一份外賣。你勇敢地寫下了 `fetch` 或 `axios`，發送了你的呼喚，也等到了回應。太棒了！

但現在，你的「外賣」已經送到了家門口。打開門，你看到一個包裹，裡面究竟裝了什麼？這就是我們今天要處理的重點：**如何拆開這個包裹，看懂裡面的東西，並把它們好好地利用起來！**

---

## 【第 17 天：處理 API 回傳資料】

### 🎯 今日目標：
1. 理解 API 回傳資料的常見格式：JSON。
2. 學會如何將原始回應轉換成 JavaScript 物件。
3. 掌握如何從這些資料結構中提取你需要的資訊。
4. 知道如何將這些資料呈現在網頁上。

---

### 一、打開包裹：認識 JSON (JavaScript Object Notation)

當你向 API 發出請求後，最常見的回傳資料格式就是 **JSON**。你可以把它想像成是 JavaScript 物件和陣列的文字表示形式。它非常輕巧、易於人類閱讀，也易於機器解析。

**JSON 長什麼樣子？**

通常，你會看到兩種基本結構：

1.  **物件 (Object)：** 用 `{}` 包裹，由鍵值對 (key-value pairs) 組成。
    ```json
    {
      "title": "第 17 天的 API 課程",
      "author": "你的程式導師",
      "duration": 60,
      "isBeginnerFriendly": true
    }
    ```

2.  **陣列 (Array)：** 用 `[]` 包裹，裡面包含多個項目，每個項目又可以是物件、字串、數字等。
    ```json
    [
      {
        "id": 1,
        "name": "香蕉",
        "price": 30
      },
      {
        "id": 2,
        "name": "蘋果",
        "price": 50
      },
      {
        "id": 3,
        "name": "橘子",
        "price": 25
      }
    ]
    ```

是不是很眼熟？這幾乎就跟你在 JavaScript 裡面寫物件和陣列一樣！這就是 JSON 如此方便的原因。

### 二、解析資料：從 Response 到 JavaScript 物件

當你使用 `fetch` 收到回應時，它還不是一個可以直接操作的 JavaScript 物件，而是一個 `Response` 物件。你需要調用它的 `.json()` 方法來將其解析。

讓我們用一個假設的 API 範例來看看：

```javascript
// 假設我們有一個簡單的 API，會回傳一篇文章的資訊
const API_URL = 'https://jsonplaceholder.typicode.com/posts/1'; // 這是一個公共的測試 API

async function getPostData() {
  try {
    const response = await fetch(API_URL); // 發出請求，等待回應

    // 檢查回應是否成功 (HTTP 狀態碼 200-299)
    if (!response.ok) {
      throw new Error(`HTTP 錯誤！狀態碼: ${response.status}`);
    }

    const data = await response.json(); // 核心步驟：將回應解析成 JavaScript 物件
    console.log("解析後的資料：", data); // 在控制台看看解析後的資料長什麼樣子！

    // 你會看到類似這樣的輸出在你的瀏覽器控制台：
    // {
    //   "userId": 1,
    //   "id": 1,
    //   "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
    //   "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut ... (略)"
    // }

  } catch (error) {
    console.error("抓取資料時發生錯誤：", error);
  }
}

getPostData();
```

**小撇步：`console.log(data)` 是你最好的朋友！**
當你第一次處理一個新的 API 時，務必先將 `data` 印出來。它是你探索未知資料結構的雷達圖！

### 三、提取寶藏：存取資料中的特定資訊

資料已經變成 JavaScript 物件或陣列了！現在，你可以像操作任何普通 JavaScript 物件和陣列一樣來存取它們了。

**範例延續：** 假設 `data` 是上面文章的物件。

```javascript
// ... 延續上一個函式內的 data

console.log("文章標題：", data.title);      // 使用點運算子存取物件屬性
console.log("文章內容：", data.body);       // 繼續存取其他屬性
console.log("作者 ID：", data.userId);    // 甚至可以存取數字型態

// 如果資料是一個陣列 (例如，回傳了多篇文章)
const ANOTHER_API_URL = 'https://jsonplaceholder.typicode.com/posts'; // 這個會回傳一個文章陣列

async function getAllPosts() {
  try {
    const response = await fetch(ANOTHER_API_URL);
    if (!response.ok) {
      throw new Error(`HTTP 錯誤！狀態碼: ${response.status}`);
    }
    const posts = await response.json(); // 這裡 posts 會是一個陣列

    console.log("所有文章：", posts);
    console.log("第一篇文章的標題：", posts[0].title); // 陣列透過索引存取，再用點運算子存取物件屬性
    console.log("第五篇文章的內容：", posts[4].body);

    // 如果要遍歷所有文章
    console.log("\n--- 所有文章標題 ---");
    posts.forEach(post => {
      console.log(`- ${post.title}`);
    });

  } catch (error) {
    console.error("抓取所有文章時發生錯誤：", error);
  }
}

getAllPosts();
```

是不是很簡單？一旦資料被解析成 JavaScript 物件，一切都變得熟悉起來了。

### 四、展示成果：將資料呈現在網頁上

光是 `console.log` 還不夠，我們想把這些有用的資訊展示給使用者看！這時候，我們就需要操作 DOM (Document Object Model) 了。

讓我們在 HTML 檔案中加入一些元素：

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 17: API 回傳資料處理</title>
</head>
<body>
    <h1>最新文章</h1>
    <h2 id="post-title">載入中...</h2>
    <p id="post-body">載入中...</p>

    <h2>所有文章列表</h2>
    <ul id="posts-list">
        <li>載入中...</li>
    </ul>

    <script src="app.js"></script> <!-- 確保你的 JavaScript 檔案名稱是 app.js -->
</body>
</html>
```

然後，在你的 `app.js` (或者你喜歡的 JavaScript 檔案) 中：

```javascript
// app.js

const API_SINGLE_POST = 'https://jsonplaceholder.typicode.com/posts/1';
const API_ALL_POSTS = 'https://jsonplaceholder.typicode.com/posts';

const postTitleElement = document.getElementById('post-title');
const postBodyElement = document.getElementById('post-body');
const postsListElement = document.getElementById('posts-list');

// 函數：顯示單篇文章
async function displaySinglePost() {
  try {
    const response = await fetch(API_SINGLE_POST);
    if (!response.ok) {
      throw new Error(`HTTP 錯誤！狀態碼: ${response.status}`);
    }
    const post = await response.json();

    postTitleElement.textContent = post.title; // 將資料設定到 HTML 元素中
    postBodyElement.textContent = post.body;

  } catch (error) {
    console.error("無法顯示單篇文章：", error);
    postTitleElement.textContent = "載入失敗！";
    postBodyElement.textContent = "請檢查網路或 API。";
  }
}

// 函數：顯示所有文章列表
async function displayAllPosts() {
  try {
    const response = await fetch(API_ALL_POSTS);
    if (!response.ok) {
      throw new Error(`HTTP 錯誤！狀態碼: ${response.status}`);
    }
    const posts = await response.json();

    // 清空現有的 "載入中..." 訊息
    postsListElement.innerHTML = '';

    posts.forEach(post => {
      const listItem = document.createElement('li'); // 創建新的列表項目
      listItem.textContent = `ID: ${post.id} - ${post.title}`; // 設定文字內容
      postsListElement.appendChild(listItem); // 將項目添加到列表中
    });

  } catch (error) {
    console.error("無法顯示文章列表：", error);
    postsListElement.innerHTML = '<li>載入文章列表失敗！</li>';
  }
}

// 頁面載入時執行這些函數
displaySinglePost();
displayAllPosts();
```

現在，當你打開 `index.html` 檔案時，你會看到網頁上的內容已經被 API 回傳的真實資料更新了！

### 🚀 總結與鼓勵

恭喜你！今天你學會了如何從 API 的回應中提取有價值的資訊，並將其展示在網頁上。這是一個非常、非常、非常重要的技能！因為所有與外部服務互動的現代網頁應用，都離不開這一環。

你已經從一個單純的請求者，變成了一個能夠「理解並利用」資訊的資料處理者。這代表你的應用程式開始有了「生命力」，能夠呈現動態的、即時的內容了！

別擔心一開始會有點混亂，特別是當你遇到複雜的 JSON 結構時。記住我們的小撇步：**永遠先 `console.log()`，然後仔細查看 API 文件**。這會是你的指路明燈。

繼續保持這份好奇心和學習的熱情！你做得太棒了！

下一個挑戰見！ 💪