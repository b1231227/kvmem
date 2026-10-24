# Qwen3.8-27B mmproj 4352×1152 shape 修正筆記

> 用途：未來用 KVMem 編譯**任何新版** ninfer 後端之前，先套用這個 patch，否則無法載入 community 的 K-quant Qwen3.8-27B mmproj。
> 配套檔：`Qwen3.8-27B-mmproj-4352-shape-fix.patch`（本目錄，byte-exact，可直接 apply）。

## 一、為什麼需要這個 fix

Qwen3.8-27B 的 vision mmproj，FFN intermediate width **原生是 4,304**。
GGUF **K-quant**（multi-block）以 **256** 為 super-block 單位；4,304 **不是 256 的倍數**（4304 ÷ 256 = 16.8），所以 stock quantizer 會「悄悄」把 FFN down-projection **留成未量化 F16**（連官方 [Q8_0 reference](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) 的 255 MiB FFN 塊都這麼做）。

**目標輕量化 mmproj：[HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF)**
- 檔名 `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf`，**331 MiB**（5.99 bpw）≈ BF16 reference（888 MiB）的 **37%**。
- 11 項視覺 benchmark 整體 **0.7458** vs BF16 **0.7493**（0.995 ratio），差距在測量 noise 內，當作未退化。
- 它把這**單一維度 pad 到 4,352 = 17 × 256**（補零行，數學上透明——被 pad 的通道由零行消滅），讓 FFN down-projection 能走 6-bit K-quant。

NInfer 的 linear dispatch 原本**沒有**這兩個 shape：
- q4：`4352 × 1152`（FFN **up-projection**，1152→4352）
- q5：`1152 × 4352`（FFN **down-projection**，4352→1152）

→ 載入這類 K-quant mmproj 時，dispatch 找不到對應 kernel → **無法 serve**（shape mismatch / 找不到 entry）。

## 二、Fix 來源（就是最小 diff，不用自己寫）

| 項目 | 值 |
|---|---|
| Commit | `b1231227/ninfer@72dbd773`（`feat(q4/q5): add 4352x1152 shape for Qwen3.8-27B mmproj`） |
| PR | Neroued/ninfer#408 |
| 規模 | **10 檔、68 行**（每個 quant 5 處） |
| Patch 檔 | `Qwen3.8-27B-mmproj-4352-shape-fix.patch`（與本檔同目錄） |

## 三、套用到哪、怎麼套

KVMem 的 ninfer 後端是 **`backends/ninfer` submodule**（指向 `iamwavecut/ninfer-all`）。
**該 submodule 沒有這個 fix**（fix 在你的 fork），所以每次取得新 ninfer 源碼後、編譯前，都要手動套一次。

```bash
# 1) 確保 ninfer submodule 已 checkout（backends/ninfer/src/ops/linear/ 要存在）
git submodule update --init backends/ninfer

# 2) 把 patch 套到 ninfer 樹（base path = backends/ninfer，-p1 去掉 a/ 前綴）
cd <kvmem-checkout>/backends/ninfer
patch -p1 < <路徑>/Qwen3.8-27B-mmproj-4352-shape-fix.patch
#   （若 ninfer 樹是 git repo，也可：git apply <路徑>/Qwen3.8-27B-mmproj-4352-shape-fix.patch）
```

> 注意：patch 路徑是相對於 **ninfer 根**（`src/ops/linear/...`），在 KVMem 裡 ninfer 根 = `backends/ninfer/`，所以 `-p1` + `cd backends/ninfer`（或 `patch -p1 -d backends/ninfer`）。

## 四、10 個檔改動清單（對照 / 手動改用；精確內容以 patch 為準）

**Q4 — shape `4352×1152`（`select_q4_n4352_k1152`）**

| 檔 | 改動 |
|---|---|
| `src/ops/linear/q4/shapes/n4352_k1152.cpp` | **新檔**（16 行）：定義 `select_q4_n4352_k1152`（依 token 數回傳 launch，與既有 `n4304_k1152.cpp` 同構） |
| `src/ops/linear/q4/shapes/unified.cpp` | 加 `select_q4_n4352_k1152_unified`（11 行） |
| `src/ops/linear/q4/q4_shapes.h` | 加宣告 ×2：`select_q4_n4352_k1152` + `select_q4_n4352_k1152_unified` |
| `src/ops/linear/q4/q4_dispatch.cpp` | 加 `ShapeEntry{4352, 1152, select_q4_n4352_k1152, select_q4_n4352_k1152_unified},` |
| `src/ops/linear/q4/sources.cmake` | 加 `"${CMAKE_CURRENT_LIST_DIR}/shapes/n4352_k1152.cpp"` |

**Q5 — shape `1152×4352`（`select_q5_n1152_k4352`）**

| 檔 | 改動 |
|---|---|
| `src/ops/linear/q5/shapes/n1152_k4352.cpp` | **新檔**（15 行）：定義 `select_q5_n1152_k4352` |
| `src/ops/linear/q5/shapes/unified.cpp` | 加 `select_q5_n1152_k4352_unified`（18 行） |
| `src/ops/linear/q5/q5_shapes.h` | 加宣告 ×2：`select_q5_n1152_k4352` + `select_q5_n1152_k4352_unified` |
| `src/ops/linear/q5/q5_dispatch.cpp` | 加 `ShapeEntry{1152, 4352, select_q5_n1152_k4352, select_q5_n1152_k4352_unified},` |
| `src/ops/linear/q5/sources.cmake` | 加 `"${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k4352.cpp"` |

## 五、驗證

- 編譯完成後，載入 K-quant mmproj：`F:\ai_models\Qwen3.6\27B\mmproj\mmproj-Qwen3.8-27B-Q5_K-MIX.gguf`
  → 應正常載入，**不再**報 shape mismatch / dispatch 找不到。
- 對照基準：目前 `F:\ai_system\kvmem`（08fd701 build）**已含**此 fix——`backends/ninfer/src/ops/linear/q4/shapes/n4352_k1152.cpp` 等 10 檔都在，可直接當範例比對。

## 六、未來編譯新版流程（一句話）

> 取得新 ninfer 源碼 → **編譯前**先 `patch -p1` 套本檔的 patch → 編譯 → 用 Q5_K-MIX mmproj 驗證載入。
