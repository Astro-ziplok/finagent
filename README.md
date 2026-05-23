# FinAgent - AI-Powered Financial Data Agent

A hackathon-style project for the **IT Application in Banking and Finance** course.

---

## Project Structure

```
finagent/
├── data/
│   ├── raw/                         ← Raw data organised by ticker and type
│   │   ├── AAPL/
│   │   │   ├── prices/              ← OHLCV price CSVs
│   │   │   ├── financials/          ← Income, balance, cashflow CSVs
│   │   │   └── news/                ← News sentiment CSVs
│   │   └── shared/
│   │       ├── commodities/         ← Commodity prices CSV
│   │       ├── news/                ← RSS and NewsAPI headlines
│   │       └── quota/               ← Alpha Vantage quota tracker
│   └── processed/                   ← Cleaned, feature-engineered data
├── src/
│   ├── collection/fetch_data.py     ← Yahoo Finance + NewsAPI + RSS feeds
│   ├── cleaning/clean_data.py       ← Cleaning pipeline + feature engineering
│   ├── visualization/charts.py      ← 8 chart types including candlestick
│   ├── fundamentals/
│   │   ├── alpha_vantage.py         ← News sentiment + company overview
│   │   ├── yfinance_fundamentals.py ← Income, balance sheet, cash flow (free)
│   │   └── quota.py                 ← Alpha Vantage daily quota tracker
│   ├── analysis/ai_analysis.py      ← Groq LLM analysis functions
│   └── export/report_exporter.py   ← PDF report generator (analysis + charts)
├── reports/
│   ├── charts/                      ← Charts organised by ticker
│   ├── analysis/                    ← AI analysis text reports + exported PDFs
│   └── data_snapshots/              ← Summary statistics CSVs
├── tests/                           ← 44 unit tests (pytest)
├── agent.py                         ← PRIMARY entry point - interactive agent
├── main.py                          ← Alternative - fixed batch pipeline
├── config.py                        ← Global settings (tickers, paths, keys)
├── requirements.txt                 ← All dependencies
└── .env.example                     ← API key template
```

---

## Quick Start

> If you have never used Python or a terminal before, follow every step in order. Do not skip any step.

### Step 1 - Make sure Python is installed

Open your terminal (search "PowerShell" on Windows) and type:

```bash
python --version
```

You should see something like `Python 3.13.x`. If you see an error, download Python from [python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"** during installation.

---

### Step 2 - Download the project

```bash
git clone https://github.com/Astro-ziplok/finagent.git
cd finagent
```

If you do not have Git, download it from [git-scm.com](https://git-scm.com/downloads).

---

### Step 3 - Create a virtual environment

A virtual environment is an isolated space where Python packages are installed for this project only. It prevents conflicts with other projects on your computer.

```bash
python -m venv venv
```

You only need to do this **once**. A new folder called `venv/` will appear in the project.

---

### Step 4 - Activate the virtual environment

You must activate the venv **every time** you open a new terminal before running the project.

```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

When activated correctly, your terminal prompt will show `(venv)` at the start:

```
(venv) PS C:\Users\YourName\finagent>
```

If you do not see `(venv)`, the venv is not active. Run the activate command again before continuing.

---

### Step 5 - Install all dependencies (first time only)

```bash
pip install -r requirements.txt
```

This installs all required Python libraries. It may take 2-5 minutes depending on your internet speed. You only need to do this once.

---

### Step 6 - Set up your API keys

```bash
copy .env.example .env        # Windows
cp .env.example .env          # Mac / Linux
```

Open the new `.env` file in VS Code and fill in your API keys:
```bash
GROQ_API_KEY=your_groq_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_newsapi_key_here   # Optional, the agent runs fine without it
```

---

### Step 7 - run the agent

```bash
python agent.py
```

You should see the FinAgent banner and a `You:` prompt. Type any financial question to begin.

---

### Every time you come back

You only need Steps 4 and 7 for every new session:

```bash
venv\Scripts\activate        # Step 4 - activate venv
python agent.py              # Step 7 - run the agent
```

---

## Using the Interactive Agent

Once running, type any natural language query:

```
You: Give me a full report on Nvidia
You: Compare Apple and Microsoft
You: Show me the income statement for Amazon 2024 vs 2025
You: What is the risk of TSLA?
You: Show me fundamentals for AAPL
You: Show me cash flow for Amazon
You: Latest news for GOOGL
You: Show me commodity prices
You: What sectors are trending right now?
You: Compare AAPL and MSFT from 2020 to 2023
You: quit
```

| Query type | Example | Alpha Vantage cost |
|---|---|---|
| Analyse | "Analyse Apple" | 0 |
| Compare | "Compare AAPL and MSFT" | 0 |
| Trend | "Show me the trend for TSLA" | 0 |
| Risk | "What is the risk of GOOGL?" | 0 |
| Full report | "Full report on Nvidia" | up to 4 per ticker |
| Fundamentals | "Fundamentals for AAPL" | 0 (yfinance) |
| Income statement | "Income statement Amazon 2024 vs 2025" | 0 (yfinance) |
| Balance sheet | "Balance sheet MSFT" | 0 (yfinance) |
| Cash flow | "Cash flow Amazon" | 0 (yfinance) |
| News | "Latest news for GOOGL" | 1 per ticker |
| Commodities | "Show me commodity prices" | 0 (Yahoo Finance) |
| Sectors | "What sectors are trending?" | 0 (Yahoo Finance) |

Type `quota` to check your remaining Alpha Vantage daily requests.
Type `help` to see query examples again.

After every analysis, the agent will ask:
```
Export this analysis as a PDF report? (yes/no):
```
Type `yes` to generate a formatted PDF containing the full analysis text and all charts
saved to `reports/analysis/TICKER_report_YYYY-MM-DD.pdf`. Requires Microsoft Word on Windows.

---

## Alternative: Fixed Pipeline (main.py)

Runs a batch analysis on the 5 default US tickers without interaction:

```bash
python main.py
```

This pipeline produces trend summaries, risk commentary, comparative analysis and charts
for AAPL, MSFT, GOOGL, TSLA, AMZN. Runtime: approximately 2 minutes.

To change the default tickers, edit `TICKERS` in `config.py`.

---

## API Keys

| Key | Where to Get | Cost |
|-----|-------------|------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free |
| `ALPHA_VANTAGE_KEY` | [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) | Free (25/day) |
| `NEWS_API_KEY` | [newsapi.org/register](https://newsapi.org/register) | Free, optional |

Add all keys to your `.env` file. Never commit `.env` to GitHub - it is listed in `.gitignore`.

---

## Unit Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

44 tests covering cleaning, collection, visualization and analysis modules.
All tests use synthetic data - no real API calls are made during testing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv first: `venv\Scripts\activate`, then `pip install -r requirements.txt` |
| Stocks not loading | Turn on a VPN - Yahoo Finance blocks some regions including Vietnam |
| Alpha Vantage returns empty | 25 requests/day limit hit. Type `quota` to check. Resets at 7 AM |
| `GROQ_API_KEY not set` | Check `.env` file exists and has no spaces around the `=` sign |
| Groq API 403 error | Network issue - turn on VPN and try again |
| Yellow warnings in VS Code | `Ctrl+Shift+P` → `Python: Select Interpreter` → select the venv option |
| `cd finagent` fails | Run `dir` to check your current folder location |
| PDF export fails | Microsoft Word must be installed. Falls back to .docx if Word is missing |