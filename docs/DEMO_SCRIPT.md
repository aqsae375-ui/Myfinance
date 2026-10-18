# Myfinance: 5-minute demo script

Uses the synthetic sample statement (Mar-Aug 2026) and only the 5 questions prepared and recorded by
`scripts/prepare_demo.py`. Everything runs offline on the demo model, so every answer below appears exactly as
written and in under a second.

## Before the session (10 minutes ahead)

1. Open PowerShell in the project folder:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   Remove-Item Env:LLM_PROVIDER -ErrorAction SilentlyContinue   # the demo must use the offline model
   python scripts/prepare_demo.py
   ```
   Expected: `9/9 passed.` If anything says FAIL, see "Recovery" below before going live.
2. `streamlit run app.py` and open http://localhost:8501 in the browser. Zoom to 110-125% for the room.
3. In the sidebar, under **Privacy**, **AI model** must read `fake - Offline demo model (keyword planner, no LLM)`
   and the green pill must say `Local - nothing leaves this machine`.
4. Keep a second browser tab open on `results/evaluation_summary.md` (on GitHub) and a terminal ready (recovery).

## The walkthrough

### 0:00-0:30 Pitch (no clicks)

Say: "Myfinance reads your bank statement and tells you where your money goes, answers questions about it, and plans
your savings. Two promises: every number is computed by code, never by the AI, and your personal details are
masked before any text reaches a model."

### 0:30-1:30 Dashboard

1. Click **Load sample data** (sidebar, blue button).
   On screen: sidebar shows `Using: Sample data (synthetic)`; the month selector at the top right shows
   **August 2026**; the **Dashboard** tab is selected.
2. Point at the four KPI cards.
   On screen: **Total Spent Rs 3,508**, **Income Rs 4,899**, **Net Saved Rs 1,391**, **Savings Rate 28.4%**, each
   with its change vs July.
3. Scroll to **What changed**.
   On screen: an "Unusual" card for `INTL WIRE TO IBAN` Rs 750 on 22 Aug, and a "Check" card saying
   `FITPRO GYM MEMBERSHIP may be unused: no related activity in 134 days, but it still costs Rs 44.99 a month`.
   Say: "These sentences are generated from computed numbers by rules. No model is involved on this page at all."
4. Open the month selector, choose **June 2026**, scroll to **Transactions**, click the pill **Needs review (1)**.
   On screen: one row dated Jun 11, Rs 9.99, with an orange **Needs review** badge.
   Say: "This description is a planted prompt injection: 'ignore previous instructions...'. Myfinance recognized it,
   never sent it to a model, and put it in the review queue for me."
5. Set the month selector back to **August 2026**.

### 1:30-3:30 Ask (the 5 prepared questions)

Click the **Ask** tab. For each question, type it exactly into the box ("Ask a question about your statement")
and click **Send**.

| # | Type exactly | Expected answer on screen | Point at |
|---|---|---|---|
| 1 | `How much did I spend on dining out in May 2026?` | "You spent **Rs 1,535.37** on Dining Out in 2026-05 across 22 transactions (35.25% of that month's spending)." Chip: `22 source transactions` | Click **✓ Verified · How I got this** under the answer: step 1 shows `get_category_spending(category="Dining Out", month="2026-05")`, step 2 the returned fields, step 3 `1,535.37 ✓ 35.25 ✓ ...` |
| 2 | `Compare my spending in May 2026 and June 2026.` | "In 2026-06 you spent **Rs 4,349.50**, down **Rs 5.80** (0.13%) from **Rs 4,355.30** in 2026-05. Net saved was Rs 541.56, up Rs 1.76 (0.33%). Biggest change: Shopping, up Rs 1,349.00 ..." Chip: `75 source transactions` | Say: "The model picked the tool. The subtraction was done by code." |
| 3 | `Which subscriptions might be unused?` | A list of 4 subscriptions; **FITPRO GYM MEMBERSHIP: Rs 44.99/month (Rs 539.88/year), possibly unused: no related activity for 134 days (last on 2026-04-16) but still charged** | Say: "A bank statement can't see whether you watch Netflix, so only a gym with no visits gets flagged." |
| 4 | `Were there any unusual charges in July 2026?` | "I found 1 unusual charge in 2026-07: 2026-07-08 CITY POWER & LIGHT (Utilities) **Rs 486.20**, 8.64x the usual Rs 56.30, Rs 429.90 above it" Chip: `1 source transaction` | Open **How I got this** -> scroll to the source row table: the one bill. |
| 5 | `Should I invest my savings in bitcoin?` | "Sorry, I can only help with questions about your own transactions in this statement: ..." | Expander label reads **Refused (out of scope) · How I got this**: "no model or tool was called". |

Say after question 5: "If a model ever writes a number that no tool returned, the verifier replaces the whole
answer with one built from the tool results. In our 24-question test with a local model, zero unverified numbers
reached the user."

### 3:30-4:30 Your Plan

1. Click the **Your Plan** tab.
   On screen: a **Set your goal** card; the other cards say "Set a goal to see this".
2. Goal type **Target by date** (selected by default). Name: `Emergency fund`. Target (Rs): `10000`.
   Deadline: **2027-03-31**. Already saved: leave empty. Click **Save goal**.
   On screen, top row: **Your goal: Save Rs 10,000 by Mar 31, 2027**; **Monthly surplus Rs 1,838** (▲ Rs 587 vs
   Rs 1,251 now); **Months to goal 6** (▼ 2 months vs 8 now); **12-month savings Rs 22,059** (vs Rs 15,016 now).
   Below it, **Ways to save, ranked**, 5 ticked actions in this order:
   1. Plan big Shopping purchases ahead (a typical month has none): Rs 255 / month
   2. Keep Dining Out to your typical month of Rs 278.21: Rs 197 / month
   3. Review the unusual Rs 486.20 CITY POWER & LIGHT charge and plan to avoid a repeat: Rs 72 / month
   4. Cancel FITPRO GYM MEMBERSHIP (Rs 44.99 a month): Rs 45 / month
   5. Keep Groceries to your typical month of Rs 598.57: Rs 19 / month
3. Untick **Cancel FITPRO GYM MEMBERSHIP (Rs 44.99 a month)**.
   On screen: **Monthly surplus Rs 1,793** (▲ Rs 542 vs Rs 1,251 now), **12-month savings Rs 21,519**, and the
   **Current path vs your plan** chart redraws. Tick it again (back to Rs 1,838).
   Say: "This is pure arithmetic on your own history, so it updates as you click."
4. Saving the goal stores it in the demo database. Before the next run, `python scripts/prepare_demo.py` resets it.

### 4:30-5:00 Close

Say: "Privacy: redaction before any model, a fully local mode, nothing stored in the cloud. Trust: every number
from code, checked before you see it. We publish our failures too." Switch to the evaluation tab: categorization
57/60 on our development set; 8/8 planted injections blocked; 0 PII leaks in every logged model call.

## Recovery (if something fails)

| Symptom | Do this | Say |
|---|---|---|
| Blank page, error box, or "Something went wrong" | Press F5. Click **Load sample data** again. | "Live demos: let me reload." |
| "Couldn't build the sample data" in the sidebar | In the terminal: `python scripts/prepare_demo.py` (about 30 s), then F5 and **Load sample data**. | "It rebuilds the demo data from scratch, offline." |
| An Ask answer differs from the table, or takes more than a few seconds | Sidebar **AI model** -> choose `fake`. Ask again. | "That was a live model; for time we'll use the offline demo model, which runs the same tools and the same verifier." |
| An answer shows **⚠ Verification failed: answer rebuilt from tool results** | Open the expander and show the ✗ number. | "This is the safety net working: the model's draft had a number no tool returned, so it was replaced." |
| The app won't start at all | In the terminal: `python scripts/chat_cli.py --demo`, then open `results/demo_traces.jsonl` and `results/evaluation_summary.md`. | "Here is the same agent in the terminal, and the recorded answers with their full traces." |
| Your Plan opens on an old goal instead of **Set your goal** (an earlier rehearsal saved one) | Click **Edit goal** on the goal card and enter the demo goal, or carry on with the goal shown. | "Here's a goal I set earlier; let me change it." |
