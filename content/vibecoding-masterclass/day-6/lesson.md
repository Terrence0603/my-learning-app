好的，未來的程式碼英雄！今天我們要一起探索一個超級有用的工具，它會讓你寫程式的過程更安全、更有條理，就像擁有了程式碼的「時光機」一樣！

---

## 【第 6 天：Part 2: 你的程式時光機！版本控制概念大揭秘 (Git、Commit、Branch)】

哈囉，各位程式碼探險家！恭喜你一路走到今天，你的程式功力肯定又提升了不少！

在之前的學習中，你可能已經寫了一些程式碼，感受到了創造的樂趣。但有沒有遇過這種情況：
*   不小心改壞了程式，想回到上一個能跑的版本，卻發現檔案已經存檔蓋掉了？
*   嘗試一個新功能，結果把舊程式搞得一團糟，最後只好砍掉重寫？
*   未來可能和朋友一起寫程式，怎麼確保大家改來改去不會互相影響？

別擔心！今天我們要介紹的「版本控制」就是為了解決這些痛點而生的超級英雄！

### 1. 什麼是版本控制？它為什麼這麼重要？

想像一下，你正在寫一本很重要的書。版本控制系統就像是一個貼心的助手：
*   每次你完成一章，它都會幫你拍一張**快照 (Snapshot)**，並記錄下你做了什麼改動。
*   如果你對某一章不滿意，想回到以前的草稿，它可以讓你輕鬆**還原 (Revert)**。
*   你可以在主線故事進行的同時，開啟一個「平行宇宙」來嘗試新的情節，而不會影響主線。
*   如果你的朋友也想幫忙寫書，它可以協調你們的修改，避免衝突。

而眾多版本控制系統中，最廣泛、最強大也最受歡迎的，就是我們的主角：**Git**！

### 2. Git：你的程式碼守護神

Git 是一個**分散式版本控制系統 (Distributed Version Control System, DVCS)**。聽起來很高大上對不對？簡單來說，就是你的程式碼歷史紀錄不只存在一個中央伺服器，而是每一個參與者（也就是你！）的電腦裡都有一份完整的副本。這讓它非常強大、可靠，而且大部分操作都可以在本地電腦上快速完成！

今天我們就來認識 Git 的三個最核心的概念：**Repository (倉庫)、Commit (提交) 和 Branch (分支)**。

### 3. Git 的核心概念：Commit (提交)

**Commit** 是 Git 世界中最基本的單位，它代表了你程式碼在某個時間點的**一個「快照」**。就像你玩遊戲時按下「存檔」一樣，每次 Commit 就會把你的程式碼現狀保存下來，並附帶一條描述你這次做了什麼改動的訊息。

**讓我們來實際操作看看吧！**

1.  **初始化一個 Git 倉庫 (Repository)**：
    首先，在你專案的資料夾裡（比如 `my_project`），告訴 Git：「從現在開始，請你幫我管理這個資料夾！」

    ```bash
    mkdir my_project
    cd my_project
    git init
    ```
    執行 `git init` 後，你會看到一個訊息，表示它已經成功初始化了一個空的 Git 倉庫。此時，你的資料夾裡會多出一個隱藏的 `.git` 資料夾，這就是 Git 儲存所有版本紀錄的地方。

2.  **建立你的第一個檔案並進行第一次 Commit**：
    我們來寫一個簡單的 Python 程式。在 `my_project` 資料夾中建立一個 `hello.py` 檔案，內容如下：

    ```python
    # hello.py
    print("哈囉，程式世界！")
    ```

    現在，你的資料夾裡有了新檔案。Git 還不知道它需要被追蹤，你可以用 `git status` 來查看現狀：
    ```bash
    git status
    ```
    你會看到 `hello.gpy` 被標示為 `Untracked files` (未追蹤的檔案)。

    **把檔案加入暫存區 (Staging Area)：`git add`**
    在進行 Commit 之前，你需要告訴 Git 哪些檔案的變更要被包含在這次快照中。這就是 `git add` 的作用，它將檔案從工作目錄移動到一個叫做「暫存區」的地方。

    ```bash
    git add hello.py
    # 或者如果你有很多改動，可以直接用 git add . 來加入所有變更
    ```
    再次使用 `git status`，你會看到 `hello.py` 現在變成了 `Changes to be committed` (等待提交的變更)。

    **提交變更：`git commit`**
    現在，是時候把暫存區裡的變更打包成一個 Commit 了！

    ```bash
    git commit -m "Initial commit: Add hello.py with a greeting message."
    ```
    `-m` 後面的字串就是你的 Commit 訊息。好的 Commit 訊息應該簡潔地說明這次 Commit 做了什麼。

3.  **修改檔案並進行第二次 Commit**：
    我們來修改 `hello.py`，增加一點功能。

    ```python
    # hello.py
    print("哈囉，程式世界！")
    print("今天是程式碼的第 6 天，我們正在學習 Git！")
    ```

    重複 `add` 和 `commit` 的步驟：
    ```bash
    git status           # 會看到 hello.py 被標示為 Modified (已修改)
    git add hello.py
    git commit -m "Feature: Add a message about Git learning."
    ```

4.  **查看 Commit 歷史：`git log`**
    你想看看你都做了哪些 Commit 嗎？

    ```bash
    git log --oneline
    ```
    你會看到類似這樣的輸出：
    ```
    <commit_hash> (HEAD -> main) Feature: Add a message about Git learning.
    <commit_hash> Initial commit: Add hello.py with a greeting message.
    ```
    每一行都是一個 Commit，最上面是最新的。那個亂碼就是每個 Commit 獨特的 ID。是不是很有成就感？你的程式碼已經有了歷史紀錄了！

### 4. Git 的另一張王牌：Branch (分支)

想像一下，你正在開發一個很重要的功能，但這個功能可能需要很多時間，而且會讓程式碼暫時變得不穩定。如果你直接在主線上開發，就可能影響其他功能的開發或導致程式無法正常運作。

這時，**Branch (分支)** 就派上用場了！它允許你在不影響主線程式碼的情況下，開闢一個「平行宇宙」來獨立開發新功能或修復 Bug。完成後，再把這個分支的成果合併回主線。

預設情況下，你所有的 Commit 都會發生在一個叫做 `main` (或 `master`) 的主分支上。

**讓我們來體驗 Branch 的魔力吧！**

1.  **查看目前的分支**：
    ```bash
    git branch
    ```
    你會看到 `* main`，星號表示你目前所在的分支是 `main`。

2.  **建立一個新分支**：
    我們來建立一個叫做 `feature/add-farewell` 的分支，用於添加告別訊息。

    ```bash
    git branch feature/add-farewell
    git branch # 再次查看，你會看到兩個分支，但你還在 main 上
    ```

3.  **切換到新分支**：
    現在，我們切換到 `feature/add-farewell` 分支，準備在這個分支上進行開發。

    ```bash
    git checkout feature/add-farewell
    git branch # 再次查看，你會看到星號移動到了 feature/add-farewell
    ```
    這時，你的程式碼文件 `hello.py` 的內容還是和 `main` 分支一樣。

4.  **在新分支上修改程式碼並 Commit**：
    修改 `hello.py`，加入告別訊息：

    ```python
    # hello.py
    print("哈囉，程式世界！")
    print("今天是程式碼的第 6 天，我們正在學習 Git！")
    print("學習 Git 很有趣！再見囉！") # 新增這行
    ```

    像之前一樣，`add` 和 `commit`：
    ```bash
    git add hello.py
    git commit -m "Feature: Add a farewell message."
    ```
    這個 Commit 只會出現在 `feature/add-farewell` 這個分支的歷史紀錄中。

5.  **切換回主分支 (main)**：
    神奇的事情要發生了！當你切換回 `main` 分支時，你會發現 `hello.py` 的內容**變回了沒有告別訊息的樣子**！

    ```bash
    git checkout main
    ```
    打開 `hello.py`，你會看到它已經恢復到你在 `main` 分支上最後一次 Commit 的狀態。這就是分支的強大之處——你可以在不同的分支上獨立工作，互相不受影響！

未來我們會學習怎麼把這些成果合併 (Merge) 回主線，但在這之前，理解分支讓你能夠「平行開發」的概念，就已經是非常大的一步了！

### 總結與鼓勵

恭喜你！今天你學會了 Git 的三大基石：
*   **Repository (倉庫)**：你的專案資料夾，被 Git 追蹤。
*   **Commit (提交)**：程式碼的快照，記錄了每次的變更。
*   **Branch (分支)**：讓你能在獨立的「平行宇宙」中開發，不影響主線。

Git 剛開始接觸可能會覺得有點抽象，但只要多練習，你會發現它真的能極大地提高你的開發效率和安全性。再也不怕不小心蓋掉檔案了！

下一堂課，我們可能會更深入地探討如何將你的本地 Git 倉庫與遠端倉庫（例如 GitHub）連接起來，實現真正的協同開發！

繼續加油！你正在成為一名更專業的開發者！🚀