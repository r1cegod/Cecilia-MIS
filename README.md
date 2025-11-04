# 🧠 Cecilia-MIS (Market Intelligence System)

**Goal:** Identify and analyze emerging market trends for digital product creation.

## 🔍 Overview
Cecilia-MIS connects to the **Google Trends API (alpha)** to collect topic growth data,  
combines it with **Reddit + YouTube signals**, and uses a **LangGraph pipeline** for:
- Trend extraction
- Pain signal mapping
- NT-LEWD scoring (Large-Early-WhoPays-Desperate)
- Data visualization and insight generation

## 🧱 Tech Stack
- **Python**, **LangGraph**, **Postgres**, **FastAPI**
- **Google Trends API** (official alpha)
- **Claude + Codex hybrid agents**

## 📊 Output
Clean CSV datasets + interactive dashboard summarizing:
| Metric | Description |
|--------|--------------|
| Trend Growth % | 90-day search increase |
| Pain Mentions | Reddit / TikTok text signals |
| NT-LEWD Score | Opportunity strength index |

## ⚙️ License
MIT License © 2025 r1cegod
