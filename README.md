
# Data Analyst Agent – Autonomous Python Executor with Image Support

A FastAPI micro-service that turns plain-English questions into answers by letting **Gemini 2.5 Flash** write, run, and debug its own Python code.  
Drop any mix of **CSV / Excel / Parquet** and **JPG / PNG / WebP** files and get back **only JSON** – no hallucinations, no hand-waving.

---

## 🚀 What it does

1. Accepts a multi-part form request:
   - `questions.txt` – the business question in plain text  
   - Any number of data files (`.csv`, `.xlsx`, `.parquet`, …)  
   - Any number of images (`.jpg`, `.png`, `.webp`, …)

2. Gemini follows a **strict two-step pipeline**:
   1. **Inspection** – list columns / describe images  
   2. **Analysis** – write **one** self-contained script that loads the real data, calculates, plots, and prints **a single JSON object**.

3. Code is executed in an **isolated temp directory** with a 120 s timeout per script and 180 s total timeout.

4. If the script crashes, Gemini is forced to fix the error; **no fabricated numbers are ever returned**.

5. When the JSON is successfully produced, Gemini replies with the word `OK` and the service returns the JSON response to you.

---

## 🔧 Tech stack

| Layer | Tech |
|-------|------|
| Orchestration | FastAPI + Uvicorn |
| LLM | Google `gemini-2.5-flash` via the **new** `google-genai` SDK |
| Sandbox | `asyncio.create_subprocess_exec` + temporary directory |
| Allowed libraries | pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, requests, beautifulsoup4, lxml, openpyxl, pyarrow, fastparquet, s3fs, Pillow, duckdb |
| Image handling | Pillow + base64 encoding |

---

##  Quick start

### 1. Clone & install

```bash
git clone https://github.com/your-org/data-analyst-agent.git
cd data-analyst-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # see below
```


### 2. Add your Google API key

```bash
echo "GOOGLE_API_KEY=YOUR_ACTUAL_KEY" > .env
```

### 3. Run

```bash
python main.py
# → Listening on http://0.0.0.0:8000
```

---

## API contract

### Endpoint
`POST /api/` – `multipart/form-data`

### Required parts
| Field | Type | Description |
|-------|------|-------------|
| `questions.txt` | text file | The question(s) to answer |
| `<any-name>.csv` | binary | Data file (can be repeated) |
| `<any-name>.jpg` | binary | Image file (can be repeated) |

### Response
- **200** – `application/json` containing **only** the JSON that the agent printed.  
- **408** – global timeout (180 s) reached.  
- **400 / 500** – malformed request or agent failure.

---

## 🖼️ Example (cURL)

```bash
curl -X POST http://localhost:8000/api/ \
  -F "questions.txt=@question.txt" \
  -F "sales.csv=@sales_data.csv" \
  -F "dashboard.png=@dashboard_screenshot.png"
```

`question.txt`
```
How many transactions occurred in December?  
Return JSON: {"december_transactions": <int>}
```

Possible response
```json
{"december_transactions": 4219}
```

---

## 🔒 Safety & limits

- Code runs under the same UID as the container/process; no network or syscall whitelisting yet – **run in a container if exposed publicly**.  
- 8 LLM iterations max, 120 s per script, 180 s total.  
- Only libraries listed above are available; pip-install attempts will fail.  
- Images are passed to Gemini’s vision model; **no pixel manipulation code is executed locally**.

---

## 🧼 Development tips

| URL | Purpose |
|-----|---------|
| `GET /health` | Lightweight liveness probe |
| Logs | Everything prints to stdout (iterations, code, stdout/stderr); set `LOG_LEVEL=debug` if needed. |



## 📄 License

MIT – feel free to embed in larger products.

