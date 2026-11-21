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
│   ├── valuation/                   ← DCF, reverse DCF, residual income, DDM, peer comparables
│   │   ├── engine.py                ← Chooses methods per company type, weights them, cites sources
│   │   ├── models.py                ← The valuation maths (pure functions, unit-tested)
│   │   ├── inputs.py                ← Treasury yield, estimates, peers (each with its source)
│   │   └── summary.py               ← Terminal valuation summary
│   └── report/
│       ├── research_report.py       ← Research report (HTML + PDF) in equity-research layout
│       └── figures.py               ← Sidebar figures for the report
├── reports/
│   ├── charts/                      ← Charts organised by ticker
│   ├── analysis/                    ← AI analysis text reports (terminal output)
│   ├── research/                    ← Research reports: TICKER_date/report.html, report.pdf, figNN.png, data.json
│   └── data_snapshots/              ← Summary statistics CSVs
├── tests/                           ← unit tests (pytest)
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

**If you do not see `(venv)`, do not continue.** Run the activate command again before proceeding.

**Windows only - if you see a red error about "running scripts is disabled on this system"**, follow these steps:

1. Close VS Code. Open **"PowerShell"** in the Window Start menu.
2. Run this command:
```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
3. Verify it worked by running:
```bash
Get-ExecutionPolicy
```
It should return `RemoteSigned`. This fix is permanent for your user account and only needs to be done once.

4. Go back to your project terminal and run the activate command again:
```bash
venv\Scripts\activate
```

---

### Step 5 - Install all dependencies (first time only)

```bash
pip install -r requirements.txt
```

This installs only the libraries the code actually uses (about 1 minute on a normal connection). You only need to do this once.

To also run the unit tests, install the dev extras instead: `pip install -r requirements-dev.txt`

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
| Full report | "Full report on Nvidia" | up to 2 per ticker |
| Fundamentals | "Fundamentals for AAPL" | up to 1 per ticker (overview), rest via yfinance |
| Income statement | "Income statement Amazon 2024 vs 2025" | 0 (yfinance) |
| Balance sheet | "Balance sheet MSFT" | 0 (yfinance) |
| Cash flow | "Cash flow Amazon" | 0 (yfinance) |
| Valuation | "What is Apple worth?" / "Is NVDA overvalued?" | 1 per day (10-year Treasury yield, cached) |
| News | "Latest news for GOOGL" | 1 per ticker |
| Commodities | "Show me commodity prices" | 0 (Yahoo Finance) |
| Sectors | "What sectors are trending?" | 0 (Yahoo Finance) |

Type `quota` to check your remaining Alpha Vantage daily requests.
Type `help` to see query examples again.

After every analysis, the agent will ask:
```
Create a research report for AAPL? (yes/no):
```
Type `yes` to build a research report laid out like an equity research note (CFA Research
Challenge style): a headline thesis and overall view, then Investment Summary, Business
Description, Market Performance & Risk Profile, Financial Analysis, Valuation and Investment
Risks, with numbered figures in a left sidebar and financial tables in the appendix.
Asking about two or more tickers (e.g. "Compare AAPL and MSFT") produces a comparison report.

Saved to `reports/research/TICKER_YYYY-MM-DD/`:

| File | What it is |
|---|---|
| `report.html` | The report. Opens automatically in your browser |
| `report.pdf` | Created automatically if Microsoft Edge or Google Chrome is installed (otherwise press Ctrl+P → Save as PDF) |
| `figNN.png` | The figures, kept as separate files so the HTML stays small |
| `data.json` | Every number the report used, handy for checking or for asking Claude about the report |

No Microsoft Word is needed. A single-company report takes about 30-80 seconds; comparisons can take up to
~2 minutes because the agent waits out the Groq free tier's per-minute token limit between AI calls.

---

## Valuation

Every research report includes a valuation section, and `"What is X worth?"` prints a valuation in the
terminal. There is **no buy/sell rating**: the report shows a model fair value, its range and the implied
upside or downside, and explains what drives it.

**Methods are chosen to fit the company** (CFA practice), rather than running every model:

| Company type | Detected by | Weighted methods | Cross-checks |
|---|---|---|---|
| Profitable, cash-generative | Positive free cash flow in most of the last 4 years | DCF (FCFF) 50% + peer multiples 50% | Reverse DCF, peer regression, own P/E history |
| Bank / insurer | Yahoo industry is a bank, insurer or capital-markets firm | Residual income 50% + peer multiples 50% | DDM (if payout >= 40%), own P/E history |
| Mature dividend payer | Dividend yield >= 2% and payout >= 40% | DDM 40% + DCF 30% + peers 30% | Reverse DCF, regression, history |
| High-growth / unprofitable | Negative or unstable free cash flow, or negative EPS | EV/Sales and P/E peer multiples | Reverse DCF where possible |

Methods that don't fit are listed with the reason they were not used (e.g. residual income for Apple,
whose book value is shrunk by buybacks).

**What each method uses:**
- **DCF (FCFF):** 10-year forecast. Years 1-2 use analyst consensus revenue growth, later years fade to
  terminal growth. Free-cash-flow margin is the 3-year average after deducting stock-based compensation.
  Discounted at WACC (CAPM cost of equity with Blume-adjusted beta). Includes bull/bear scenarios and a
  WACC x terminal-growth sensitivity grid.
- **Reverse DCF:** the annual revenue growth today's price requires, compared with analyst consensus.
- **Residual income / DDM:** analyst EPS estimates, book value and dividends, discounted at the cost of equity.
- **Peer multiples:** forward P/E, EV/EBITDA, EV/Sales or P/B (depending on company type) at the peer median
  (range: 25th-75th percentile), plus a regression of peers' P/E on expected growth (P/B on ROE for banks).

**Every assumption is cited** in the report (sidebar summary + Appendix G with full sources) and in the
terminal output. Defaults live in `config.py` and can be edited:

| Setting | Default | Where |
|---|---|---|
| Risk-free rate | Live 10-year Treasury yield (Federal Reserve H.15 via Alpha Vantage; Yahoo ^TNX fallback) | fetched, cached daily |
| Equity risk premium | 5.0% | `VALUATION_ASSUMPTIONS` |
| Terminal growth | 2.5% | `VALUATION_ASSUMPTIONS` |
| Forecast horizon | 10 years | `VALUATION_ASSUMPTIONS` |
| Beta adjustment | 0.67 x raw + 0.33 (Blume / Bloomberg) | `VALUATION_ASSUMPTIONS` |
| Method weights | see table above | `VALUATION_WEIGHTS` |
| Peer groups | Same Yahoo industry; widened to sector if fewer than 3; hand-picked overrides with reasons | `INDUSTRY_PEERS`, `TICKER_PEERS` |

**Read the results with care.** At a ~5% risk-free rate, textbook models often value mega-cap stocks well
below their market price. The reverse DCF is usually the most useful line: it shows what growth the price
already assumes. Model values are estimates that depend on the assumptions above, not forecasts.

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
| `ALPHA_VANTAGE_KEY` | [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) | Free (25/day). Used for the 10-year Treasury yield (valuation risk-free rate, 1 call/day), news sentiment and company overview |
| `NEWS_API_KEY` | [newsapi.org/register](https://newsapi.org/register) | Free, optional |

Add all keys to your `.env` file. Never commit `.env` to GitHub - it is listed in `.gitignore`.

---

## Unit Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Tests cover cleaning, collection, visualization and analysis modules, including the financial
metric calculations (Sharpe ratio, calendar quarters, benchmark stats, unit conversions).
All tests use synthetic data - no real API calls are made during testing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv first: `venv\Scripts\activate`, then `pip install -r requirements.txt` |
| Stocks not loading | Turn on a VPN - Yahoo Finance blocks some regions including Vietnam |
| Alpha Vantage returns empty | 25 requests/day limit hit. Type `quota` to check. The local counter resets on the next calendar day |
| `GROQ_API_KEY not set` | Check `.env` file exists and has no spaces around the `=` sign |
| Groq API 403 error | Network issue - turn on VPN and try again |
| Yellow warnings in VS Code | `Ctrl+Shift+P` → `Python: Select Interpreter` → select the venv option |
| `cd finagent` fails | Run `dir` to check your current folder location |
| No `report.pdf` created | Edge or Chrome was not found. Open `report.html` and press Ctrl+P → Save as PDF, or set `FINAGENT_BROWSER` in `.env` to your browser's path |

---

## Methodology Notes

How the key numbers are calculated, so results can be checked:

| Metric | Definition |
|---|---|
| Sharpe ratio | Annualised: `(mean daily return - daily risk-free rate) / daily std * sqrt(252)`. Risk-free rate is `RISK_FREE_RATE` in `config.py` (default 4%). Rough guide: <0.5 weak, 0.5-1 moderate, 1-2 good, >2 very strong |
| Volatility | Annualised: daily standard deviation of returns * sqrt(252) |
| Benchmark | Every stock is compared with `BENCHMARK_TICKER` (default SPY) over the same dates: excess return, beta and correlation |
| Quarterly breakdown | Calendar quarters (e.g. 2025-Q3); each quarter's return is measured from the previous quarter's close. Partial quarters are flagged |
| Period high/low | Called "52-week" only when the data covers about one year; otherwise labelled with the actual date range |
| Debt-to-equity | Total debt / shareholders' equity, shown as a multiple (e.g. 1.34x). Not shown when equity is negative |
| EPS growth | Forward EPS vs trailing EPS, i.e. growth analysts *expect*, not historical growth |
| Revenue growth | Most recent quarter vs the same quarter one year earlier (Yahoo `revenueGrowth`) |
| Dividend yield | Annual dividend rate / current price |
| News tone (RSS) | Simple whole-word keyword tags on headlines; directional only, not a sentiment model |

Comparisons, rankings and "increased/decreased" wording are computed in Python and handed to the
LLM as facts, so the written analysis cannot get the direction of a change wrong. The
"Background Context" section is the language model's general knowledge, not live news.

**Disclaimer:** FinAgent is an educational project. Its output is not investment advice.
