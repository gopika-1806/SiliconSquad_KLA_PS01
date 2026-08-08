# MemoryVerse AI — Digital Identity System

An AI-powered system that turns a student's scattered documents — certificates, resumes, project reports, internship letters, portfolios — into a structured, searchable, connected digital identity.

## Problem

Students accumulate proof of their growth across years: certificates, GitHub repos, internship offers, achievements. This evidence sits scattered across folders, emails, and drives. Traditional storage can save the files, but it can't understand how they connect to a person's journey.

## What it does

1. **AI Data Ingestion** — accepts uploaded files or pasted content (certificates, resumes, project reports, internship letters, portfolio links).
2. **Intelligent Categorization** — an LLM call automatically classifies each item into Projects, Skills, Certifications, Internships, Achievements, or Academics — no manual sorting.
3. **Relationship Engine** — the same call detects links to existing items (Certification → Skill → Project → Internship), so the system understands cause and effect in a person's growth, not just isolated files.
4. **Digital Journey Timeline** — every item is placed on a year-by-year timeline, visualizing growth over time.
5. **Smart Retrieval System** — a natural-language search bar ("show all my certificates", "what AI projects have I done?") is answered by sending the full item catalog to the model and asking it to pick and explain the relevant matches. Original content is never rewritten, only indexed.

## How the AI is used

Every "understanding" step (categorize, connect, search) is a single call to Claude (`claude-sonnet-4-6`) with a strict JSON-only system prompt, so the app stays deterministic and easy to reason about. There is no manual sorting anywhere in the flow — the model is the classifier, the relationship-mapper, and the retrieval layer.

## Tech

- Single-page React UI (see `MemoryVerse.jsx`)
- Claude API (`/v1/messages`) for classification, relationship-mapping, and semantic search
- No backend/database required for the demo — items live in-memory for the session; a production version would persist to a vector database (see `architecture.mermaid`)

## Running it

This is built as a self-contained React artifact. Drop it into any environment that can render a React component with `fetch` access to `api.anthropic.com/v1/messages` (e.g. Claude.ai artifacts, or a Vite/CRA app with the same API wiring).

## Files in this submission

- `MemoryVerse.jsx` — working prototype (UI + AI calls)
- `architecture.mermaid` — AI workflow / system architecture diagram
- `thought_process.md` — design reasoning, trade-offs, and what a production version would add
