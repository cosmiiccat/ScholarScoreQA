
<p align="center">
  <img src="assets/hero2.png" style="width: 100%; max-width: 1000px;" />
</p>

# 📘 ScholarScoreQA: Reliable Scholarly Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/LLM-QA-blue" />
  <img src="https://img.shields.io/badge/Retrieval-K--Span-green" />
  <img src="https://img.shields.io/badge/License-MIT-orange" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-purple" />

  <img src="https://img.shields.io/github/last-commit/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/contributors/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/forks/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/stars/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/issues/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/issues-pr/cosmiiccat/ScholarScoreQA" />

  <img src="https://img.shields.io/github/languages/top/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/languages/count/cosmiiccat/ScholarScoreQA" />
  <img src="https://img.shields.io/github/repo-size/cosmiiccat/ScholarScoreQA" />
</p>


A **retrieval-augmented scholarly question-answering system** designed for long research articles. ScholarScoreQA integrates:

* **K-Span Select** — a span-level retrieval mechanism that filters only the most relevant evidence.
* **Multi-dimensional evaluation** using a **Language Judge**, **Tone Judge**, and a unified metric called **ScholarScore**.
* **Multiple prompting strategies** (Zero-shot, CoT, Few-shot, Meta-prompting) to generate diverse answer candidates.

This repository accompanies the paper:

> **K-Span Select and Multi-Dimensional Judging for Reliable Scholarly Question Answering**
> *Preetam Pati, Sayan De, Saurabh Tiwari, Imon Mukherjee, Debarshi Kumar Sanyal*
> *IIIT Kalyani & IACS Kolkata*

---

## 🚀 Overview

Long scholarly documents present challenges for QA:

* Evidence is often **scattered** across sections.
* Retrieval systems can fetch **irrelevant text**, confusing LLMs.
* LLM-generated answers may be **fluent but hallucinated**.

**ScholarScoreQA** addresses this through:

1. **Span-level Context Engineering** via K-Span Select.
2. **Diverse LLM prompting** to generate candidate answers.
3. **Two independent judge modules** to ensure factual and stylistic reliability.
4. **ScholarScore**, a harmonic metric to pick the best final answer.

---

📘 **Repo Structure (Auto-updated)**  
See **REPO_TREE.txt** in the root directory.