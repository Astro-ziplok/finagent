# FinAgent - AI-Powered Financial Data Agent

A hackathon-style project for the **IT Application in Banking and Finance** course.

---

## Project Structure

```
finagent/
├── data/
│   ├── raw/                         ← Raw data from APIs
│   └── processed/                   ← Cleaned, feature-engineered data
├── src/
│   ├── collection/fetch_data.py     ← Yahoo Finance + NewsAPI + RSS feeds
│   ├── cleaning/clean_data.py       ← Cleaning pipeline + feature engineering
│   ├── visualization/charts.py      ← 8 chart types including candlestick
│   ├── fundamentals/
│   │   ├── alpha_vantage.py         ← News sentiment + company overview
│   │   ├── yfinance_fundamentals.py ← Income, balance sheet, cash flow (free)
│   │   └── quota.py                 ← Alpha Vantage daily quota tracker
│   └── analysis/ai_analysis.py      ← Groq LLM analysis functions
├── reports/
│   ├── charts/                      ← Charts organised by ticker
│   ├── analysis/                    ← AI analysis text reports
│   └── data_snapshots/              ← Summary statistics CSVs
├── tests/                           ← 44 unit tests (pytest)
├── agent.py                         ← PRIMARY entry point — interactive agent
├── main.py                          ← Alternative — fixed batch pipeline
├── config.py                        ← Global settings (tickers, paths, keys)
├── requirements.txt                 ← All dependencies
└── .env.example                     ← API key template
```

---

## Quick Start

Run these three commands every session:

```bash
# 1. Navigate into the project
cd finagent

# 2. Activate virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 3. Run the agent
python agent.py
```

First time only - install dependencies and set up API keys:

```bash
pip install -r requirements.txt
cp .env.example .env         # then open .env and fill in your keys
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