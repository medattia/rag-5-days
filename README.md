# Football Rules RAG, in 5 days

A RAG assistant that answers questions about football rules, upgraded with one modern technique per day and scored on the same 30-question exam.

**Documents:** IFAB Laws of the Game 2026/27, UEFA Champions League Regulations 2026/27, UEFA Disciplinary Regulations, FIFA RSTP. Download them from the official sites into `data/`.

## Progress
| Day | Technique | Answer correct (all) |
|---|---|---|
| 1 | Baseline RAG | 0.92 |
| 2 | Contextual retrieval | 0.94 |
| 3 | Hybrid search + reranking | 0.87 |
| 4 | Corrective RAG | 0.98 || 5 | Agentic RAG | 0.88 |

**Result:** Corrective RAG (Day 4) gave the most correct answers (0.98). Agentic RAG gave the most grounded answers (faithfulness 0.92), but its sub-questions often lost context.

## Run
pip install -r requirements.txt
python src/ingest.py
python src/evaluate.py --run day1_baseline
