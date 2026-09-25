# Football Rules RAG, in 5 days

A RAG assistant that answers questions about football rules, upgraded with one modern technique per day and scored on the same 30-question exam.

**Documents:** IFAB Laws of the Game 2026/27, UEFA Champions League Regulations 2026/27, UEFA Disciplinary Regulations, FIFA RSTP. Download them from the official sites into `data/`.

## Progress
| Day | Technique | Answer correct (all) |
|---|---|---|
| 1 | Baseline RAG | 0.92 |

## Run
pip install -r requirements.txt
python src/ingest.py
python src/evaluate.py --run day1_baseline