# LLM Evaluation & Regression Testing Platform

> A production-grade framework for preventing silent quality regressions in Large Language Model (LLM) systems.

---

## ❓ Problem Statement

Large Language Models are **probabilistic systems**.
They do not fail loudly when prompts, models, or configurations change.

Common failure modes include:
- Increased hallucinations after prompt updates
- Broken output formats after model upgrades
- Silent relevance degradation
- Non-deterministic behavioral drift

Traditional unit tests are **insufficient** for LLM-based systems.

This project addresses that gap.

---

## 🎯 What This Project Does

This platform enables **eval-driven development** for LLM systems by providing:

- Prompt and model versioning
- Golden datasets for deterministic testing
- Automated offline evaluation
- Regression detection against quality baselines
- Experiment tracking and auditability
- Human-readable dashboards for decision-making

The goal is **safe iteration and deployment** of LLM systems.

---

## 🧠 Design Principles

- **Model-agnostic**: Works with open-source and closed models
- **Offline-first**: No dependency on paid APIs
- **Deterministic evaluation** of probabilistic systems
- **Metrics over vibes**
- **Reproducibility over convenience**

---

## 🧰 Tech Stack

- **LLMs**: Open-source models via Ollama
- **Evaluation**: Custom Python evaluators
- **Experiment Tracking**: MLflow
- **Storage**: SQLite / PostgreSQL
- **Dashboard**: Streamlit
- **CI/CD**: GitHub Actions (planned)

---

## 📂 Repository Structure

llm-eval-platform/
├── data/ # Golden datasets
├── prompts/ # Versioned prompts
├── models/ # LLM inference wrappers
├── evals/ # Evaluation metrics & regression logic
├── tracking/ # Experiment tracking integrations
├── dashboard/ # Streamlit UI
├── runs/ # Run configs & outputs
└── scripts/ # CLI entry points


---

## 🚫 What This Is NOT

- Not a chatbot
- Not a prompt playground
- Not a demo project
- Not model benchmarking for leaderboards

This is **LLM infrastructure**.

---

## 🧪 Project Status

**Phase 0 — Foundation**
- [x] Repository scaffold
- [x] Project vision & principles

Next:
- Phase 1: Golden datasets & prompt versioning

---

## 📌 Why This Matters

LLM systems increasingly power critical workflows.
Without regression testing, failures are discovered by users.

This project demonstrates how to build **reliable, testable, and production-safe LLM platforms**.

# llm-eval-platform

## 🧪 Project Status

**Phase 0 — Foundation**
- [x] Repository scaffold
- [x] Project vision & principles

**Phase 1 — Golden Dataset & Prompt Versioning**
- [x] Golden dataset with deterministic test cases
- [x] Prompt versioning with metadata
- [x] Dataset immutability guarantees

**Phase 2 — LLM Inference Engine**
- [x] Model-agnostic inference layer
- [x] Open-source LLM integration via Ollama
- [x] Reproducible run configurations

**Phase 3 — Evaluation Framework**
- [x] Faithfulness evaluation
- [x] Relevance evaluation
- [x] Structured format validation
- [x] Offline, deterministic scoring

**Phase 4 — Experiment Tracking**
- [x] MLflow-based experiment tracking
- [x] Metric aggregation and logging
- [x] Artifact-level auditability

**Phase 5 — Regression Detection**
- [x] Baseline vs candidate comparison
- [x] Threshold-based FAIL / WARN / PASS decisions
- [x] Policy-driven regression rules

**Phase 6 — Dashboard**
- [x] Streamlit-based evaluation dashboard
- [x] Aggregate and per-sample metric visualization
- [x] Regression policy transparency
