# python-for-ai-ml

---
A free, self-contained textbook for learning Python, data science, machine learning, and AI — from first principles to production. Every chapter runs in Google Colab with no local setup required. The entire book is built around a single real-world dataset: the Stack Overflow 2025 Developer Survey.

Quick Start
Click the badge above to open the interactive Table of Contents, or jump directly to any chapter using the links below. All notebooks run in Google Colab — a free Google account is all you need.
No installation. No configuration. Open a notebook and run.

What's in the Book
The book is structured in four parts across 13 chapters and 9 appendices.
Part 1 — Core Python Fundamentals (Chapters 0–2)
Build a complete Python foundation from scratch. No prior experience assumed.
Part 2 — Data Science Foundations (Chapters 3–4)
Master NumPy, Pandas, and the Python visualisation stack.
Part 3 — Machine Learning and Artificial Intelligence (Chapters 5–11)
Build and interpret ML and AI models across three data modalities: tabular, text, and images. Covers responsible AI and adversarial robustness.
GPU recommended for Chapters 7, 8, and 9. In Colab: Runtime → Change Runtime Type → T4 GPU
Part 4 — Production and Deployment (Chapter 12)
Take a trained model from notebook to production.
Appendices
Nine reference appendices covering local environment setup, Keras, SQL, Git, Docker, security, and more.
What Makes This Book Different
One dataset, start to finish. The Stack Overflow 2025 Developer Survey runs through every chapter as a continuous project thread; you're not switching to toy examples every few pages. By Chapter 12, you've built, audited, secured, and deployed the same salary prediction system you started in Chapter 6.
Depth where most books skip. Most ML textbooks end at model training. This one covers what comes after: fairness auditing (Chapter 10), adversarial robustness and red teaming (Chapter 11), MLOps and drift monitoring (Chapter 12), supply chain security (Appendix H), and a dedicated troubleshooting guide for the bugs that don't crash your code but ruin your models (Appendix I).
Built for active learning. Every chapter contains concept-check questions with collapsible answers (69 total), and three coding exercises at progressive difficulty levels; guided scaffolds, applied challenges, and open-ended extensions (39 total, all using the SO 2025 dataset).
Zero setup. Everything runs in Google Colab. Free T4 GPU access covers all deep learning chapters.

Who This Is For
BackgroundWhere to startNo programming experienceChapter 0Some Python, new to data scienceChapter 3Python + data science, new to MLChapter 6ML background, learning deep learningChapter 7Want NLP / transformers / RAGChapter 8Want computer visionChapter 9Preparing for ML engineering interviewsChapters 6, 7, 11, 12Focused on AI safety and responsible AIChapters 10, 11, Appendix H

Prerequisites

A Google account (for Google Colab)
No local Python installation required
Chapters 0–5: no prior programming assumed
Chapters 6–12: basic Python familiarity helpful (Chapters 1–2 cover this)

The Dataset

All chapters use the Stack Overflow 2025 Developer Survey — a real-world dataset of ~15,000 curated responses covering programming languages, salaries, tools, education, and demographics. The dataset is included in the /data directory and loaded automatically in each notebook.

Suggested Learning Paths

Complete beginner (53–68 hours): Chapters 0 → 12 in order, plus appendices as needed.
Python-literate, new to ML (30–40 hours): Start at Chapter 3, skim Chapters 1–2 as reference.
ML practitioner adding depth (15–20 hours): Chapters 7–12 + Appendices H and I.
Responsible AI focus: Chapters 6, 10, 11 + Appendix H.
Production / MLOps focus: Chapters 6, 7, 12 + Appendices G, H, I.

Contributing

Found a bug, a broken Colab link, or a better explanation? Issues and pull requests are welcome. Please open an issue before submitting a large pull request so we can discuss the change first.

---

## ⚖️ Licensing
You are free to use, share, and adapt this material for any purpose, including commercial use, with attribution.

This repository uses a dual-licensing model to ensure the best experience for both readers and developers:

* **Book Content & Prose:** All written chapters, explanations, and documentation are licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](LICENSE-CONTENT).
* **Python Code Examples:** All source code, scripts, and code snippets are licensed under the [MIT License](LICENSE-CODE).

### Why this matters
I want you to be able to use the code from this book in your own projects! The **MIT License** allows for seamless code reuse. However, the **CC BY-SA 4.0** license ensures that the book's narrative and educational structure remain open and attributed to the original work if shared or adapted.
