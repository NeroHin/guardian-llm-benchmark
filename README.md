# guardian-llm-benchmark

以 YAML 配置驅動的 Guardian / safety model benchmarking framework。  
目前專注在 `pii_binary` 分類任務，目標是讓你可以：

- 用 YAML 註冊模型，不改程式碼
- 用 YAML 定義資料集路徑、內容欄位、標籤規則
- 用 YAML 組合 benchmark suite
- 用統一的 strict JSON 輸出與分類指標進行比較

## 1. 安裝

建議使用 `uv`：

```bash
uv venv
uv sync --all-groups
```

如果你用傳統方式：

```bash
uv pip install -r requirements.txt
```

OpenRouter 模型需要 API key，根目錄 `.env` 內至少放：

```bash
OPENROUTER_API_KEY=你的金鑰
```

## 2. 專案結構

```text
benchmarking/
  cli.py
  core.py
  runtime.py
  config/
  datasets/
  reporting/
  tasks/

configs/
  models/
  datasets/
  benchmarks/
```

目前預設配置：

- `configs/models/default.yaml`
- `configs/datasets/pii_datasets.yaml`
- `configs/benchmarks/pii_baseline.yaml`

## 3. 核心概念

### Models

模型只描述「怎麼呼叫」，不描述 benchmark 邏輯。

必要欄位：

- `key`
- `model_id`
- `provider`

常用可選欄位：

- `profile`
- `params_b`
- `settings`

範例：

```yaml
version: 1
models:
  - key: granite-micro-hf
    model_id: ibm-granite/granite-4.0-h-micro
    provider: huggingface
    profile: granite_guard_json
    params_b: 3.0
    settings:
      trust_remote_code: true
      torch_dtype: auto
      device_map: auto
      max_new_tokens: 256
      do_sample: false
```

### Datasets

每份 dataset 必須定義：

- `path`
- `format`
- `content_column`
- `ground_truth`

#### 固定標籤

適合整份檔案都是正樣本或負樣本：

```yaml
ground_truth:
  mode: fixed
  value: true
```

#### 欄位映射標籤

適合單一檔案同時有正負樣本：

```yaml
ground_truth:
  mode: column
  column: label
  positive_values: ["pii", 1, true]
  negative_values: ["non_pii", 0, false]
```

完整範例：

```yaml
version: 1
datasets:
  - key: multipriv_pii_positive
    task: pii_binary
    format: csv
    path: dataset/data_person_1000_target.csv
    content_column: naturalParagraph
    ground_truth:
      mode: fixed
      value: true

  - key: multipriv_pii_negative
    task: pii_binary
    format: csv
    path: dataset/data_person_1000_non_pii_100.csv
    content_column: naturalParagraph
    ground_truth:
      mode: fixed
      value: false
    filters:
      - column: generation_status
        equals: ok
```

### Benchmarks

一份 benchmark 定義：

- 要跑哪個 task
- 用哪組 dataset
- 挑哪些模型
- runtime 參數
- 輸出位置

範例：

```yaml
version: 1
benchmark:
  key: pii-baseline
  task: pii_binary
  dataset:
    positive: multipriv_pii_positive
    negative: multipriv_pii_negative
  models:
    include_keys:
      - qwen3guard06b-stream
      - granite-micro-hf
  runtime:
    sample_limit: 50
    shuffle: true
    random_seed: 42
    fail_fast: false
  outputs:
    dir: results/pii-baseline
    save_rows: true
    save_metrics_json: true
```

## 4. CLI 使用方式

列出可用配置：

```bash
uv run guardian-benchmark list models
uv run guardian-benchmark list datasets
uv run guardian-benchmark list benchmarks
```

驗證 benchmark 配置：

```bash
uv run guardian-benchmark validate --benchmark configs/benchmarks/pii_baseline.yaml
```

執行 benchmark：

```bash
uv run guardian-benchmark run --benchmark configs/benchmarks/pii_baseline.yaml
```

關閉進度列：

```bash
uv run guardian-benchmark run --benchmark configs/benchmarks/pii_baseline.yaml --no-progress
```

## 5. 輸出內容

每次 run 會輸出到：

```text
results/<benchmark-key>/<run-id>/
  rows/<model-key>.csv
  metrics/<model-key>.json
  leaderboard.csv
  run_manifest.json
```

### `rows/<model-key>.csv`

至少包含：

- `content`
- `ground_truth`
- `sample_type`
- `raw_output`
- `parsed_prediction`
- `parsed_label`
- `parsed_confidence`
- `parse_status`
- `parse_error`
- `latency_ms`
- `cost_usd`

### `metrics/<model-key>.json`

目前 `pii_binary` 會輸出：

- `accuracy`
- `precision`
- `recall`
- `f1`
- `tp`
- `tn`
- `fp`
- `fn`
- `unparseable`
- `tpr`
- `fpr`
- `overhead_latency_ms_avg`
- `overhead_latency_ms_p95`
- `cost_usd_total`
- `cost_usd_avg`

## 6. 新增 benchmark 的流程

### 新增一個模型

1. 編輯 `configs/models/default.yaml`
2. 新增一個 `models` block
3. 填入 `key / model_id / provider`
4. 視需要補 `profile / settings / params_b`

### 新增一個 dataset

1. 編輯 `configs/datasets/*.yaml`
2. 填入 `path / format / content_column`
3. 定義 `ground_truth.mode`
4. 如果需要，補 `filters`

### 新增一個 benchmark suite

1. 在 `configs/benchmarks/` 新增 YAML
2. 指定 `task`
3. 指定 `dataset`
4. 指定 `models`
5. 指定 `outputs.dir`

## 7. 嚴格 JSON 規則

`pii_binary` 的正式評分要求模型只輸出單一 JSON object：

```json
{
  "contains_pii": true,
  "label": "是",
  "confidence": 0.93,
  "reason": "包含身份證資訊"
}
```

正式 parser 只接受 strict JSON。  
非 JSON、schema 不符、欄位不一致，都會被記成 `unparseable`。

## 8. 效能與 GPU 成本建議

如果你是租 GPU server 跑 benchmark，先注意這幾點：

### 已經處理掉的瓶頸

- OpenRouter 路徑已支援 async concurrency
- benchmark core 不再透過舊版 `execution.py` 動態載入
- row-level 結果不再做重複 parsing
- Qwen stream debug log 預設關閉，避免大量 token 級寫檔

如果要開 Qwen debug：

```bash
GUARDIAN_BENCHMARK_QWEN_DEBUG=1 uv run guardian-benchmark run --benchmark configs/benchmarks/pii_baseline.yaml
```


- OpenRouter 模型：調高 concurrency，但先看 provider rate limit
- HF 本地模型：優先選單模型單卡跑滿，不要同機混跑多模型
- 若你要追求吞吐量，Granite/Qwen instruct 類分類模型會比 stream moderation 更省 GPU hours
- 大量跑 benchmark 時預設加上 `--no-progress`
- 對本地模型先用小樣本 smoke test，確認 tokenizer / remote code / 顯存配置正確再放大

## 9. 測試

```bash
uv run pytest
```
