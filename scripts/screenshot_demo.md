# Capturing the poster screenshots

Manual steps (no browser automation). Takes about 10 minutes.

## Setup
1. `.\.venv\Scripts\Activate.ps1`, then `$env:LLM_PROVIDER='fake'` (the pill reads "Local").
2. `streamlit run app.py`. The app opens at http://localhost:8501.
3. Use a 1600x1000 browser window at 100% zoom (Chrome DevTools > device toolbar > Responsive works).
   Hide the bookmarks bar. Close the sidebar (`<` arrow) for every shot except #1.
4. Click **Load sample data** in the sidebar.
5. Theme follows the browser/OS setting (the Streamlit menu is hidden). For Dark: Chrome DevTools >
   `Ctrl+Shift+P` > "Emulate CSS prefers-color-scheme: dark", then reload. Take shot #8 again in Dark.
6. Capture with Windows `Win + Shift + S` (rectangle), or DevTools > `Ctrl+Shift+P` > "Capture full size screenshot"
   for a full-page shot. Save as PNG.

## Shots
| # | View | How | What it shows |
|---|------|-----|---------------|
| 1 | Sidebar + empty state | Before step 4, sidebar open | Upload, sample data, Privacy panel "Local - nothing leaves this machine" |
| 2 | Dashboard hero | Month = **May 2026**; crop header + KPI row | 4 KPI cards, red Total-spent pill (heavy dining month), sparklines |
| 3 | Categories | Month = May 2026; crop "Where the money went" | Donut with total in the centre + this vs last month bars (Dining Out jump) |
| 4 | Trend + insights | Month = **July 2026** | Trend line with the ▲ unusual-charge marker; insight cards with "Fix this" |
| 5 | Who and when | Month = July 2026 | Top merchants + daily bars (highest day labelled: rent on 01 Jul; the 08 Jul power bill stands out) |
| 6 | Ask: proof | Ask view > click "How much did I spend on dining in May?" > open **How I got this** | Answer, green "verified" badge, tool call, arguments, result, source rows |
| 7 | Ask: safety | Ask "What stocks should I buy?" | Out-of-scope refusal badge (no model call) |
| 8 | Your Plan | Your Plan view > Save goal (defaults: Rs 10,000 by 2027-03-31) | Ranked cards with "How I got this" calculations, Months to goal 8 mo → 6 mo, before/after chart |
| 9 | Plan is live | Untick suggestion 1 (Shopping) | Chart and "With your plan" number drop instantly |
| 10 | Fix this | Dashboard (July) > click "Fix this" on the unusual-charge insight | Plan opens with "From your dashboard" banner and the highlighted card |

## Tips
- Before each shot, hover away from charts so no tooltip is showing (unless you want one: a tooltip on the
  donut makes a nice detail shot).
- After shot 8, click **Clear goal** to reset.
- For a "Local" pill shot with a real model, pick `ollama` in the sidebar's AI model box; the Dashboard does not
  change, because it never uses a model.
