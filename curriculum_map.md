# Curriculum Map — Python for AI/ML
### *A Complete Learning Journey: From Absolute Beginner to AI/ML Practitioner*

> **How to use this map:**
> Each chapter lists its learning objectives, prerequisites, key concepts, estimated time,
> and what it unlocks. Use it to plan your learning path, skip chapters you already know,
> or identify gaps before jumping in.

---

## Quick Reference

| Chapter | Title | Part | Time | GPU? | Depends on |
|---------|-------|------|------|------|------------|
| 0  | Orientation & Setup | 1 | 1–2 hrs | No | — |
| 1  | Python Fundamentals | 1 | 4–5 hrs | No | Ch 0 |
| 2  | Intermediate Python | 1 | 5–6 hrs | No | Ch 1 |
| 3  | NumPy and Pandas | 2 | 5–6 hrs | No | Ch 2 |
| 4  | Data Visualisation | 2 | 3–4 hrs | No | Ch 3 |
| 5  | SciPy & Statistics | 3 | 4–5 hrs | No | Ch 4 |
| 6  | scikit-learn | 3 | 6–7 hrs | No | Ch 5 |
| 7  | Deep Learning (PyTorch) | 3 | 5–6 hrs | **Yes** | Ch 6 |
| 8  | NLP & Transformers | 3 | 5–6 hrs | **Yes** | Ch 7 |
| 9  | Computer Vision | 3 | 5–6 hrs | **Yes** | Ch 7 |
| 10 | Ethics & Responsible AI | 3 | 4–5 hrs | No | Ch 6 |
| 11 | MLOps & Production | 4 | 5–6 hrs | No | Ch 6 |

**Total estimated time:** 53–68 hours of active work
**GPU chapters:** Enable T4 GPU in Colab — `Runtime → Change Runtime Type → T4 GPU`

---

## Part 1 — Core Python Fundamentals

### Chapter 0 — Orientation & Setup

| Property | Detail |
|----------|--------|
| **File** | `CH00_Orientation_and_Setup.ipynb` |
| **Time** | 1–2 hours |
| **Prerequisites** | None |
| **Background assumed** | None — this is the starting point |

**Learning Objectives**
- Navigate the Google Colab interface confidently
- Mount Google Drive and manage notebook persistence
- Enable GPU and TPU runtimes
- Install packages with `pip` and verify installations
- Understand the Stack Overflow 2025 dataset that runs through the entire book
- Set up a local Python environment (optional: conda, venv, VS Code)

**Key Concepts Introduced**
`!pip install` · Colab cells · Drive mounting · `import` · dataset preview

**Unlocks:** Everything — complete this before any other chapter

---

### Chapter 1 — Python Fundamentals

| Property | Detail |
|----------|--------|
| **File** | `CH01_Python_Fundamentals.ipynb` |
| **Time** | 4–5 hours |
| **Prerequisites** | Chapter 0 |
| **Background assumed** | None — starts from scratch |

**Learning Objectives**
- Declare variables and understand Python's dynamic typing
- Manipulate strings with slicing, formatting, and f-strings
- Control program flow with `if/elif/else`, `for`, `while`, `break`, `continue`
- Work with lists, tuples, sets, and dictionaries
- Write reusable functions with positional args, keyword args, `*args`, `**kwargs`, and lambda
- Apply list comprehensions for concise transformations

**Key Concepts Introduced**
Variables · data types · control flow · data structures · functions · comprehensions

**Project:** Pure-Python salary analysis on 20 SO 2025 rows — no libraries, just logic

**Unlocks:** Chapter 2

---

### Chapter 2 — Intermediate Python

| Property | Detail |
|----------|--------|
| **File** | `CH02_Intermediate_Python.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 1 |
| **Background assumed** | Basic Python syntax from Chapter 1 |

**Learning Objectives**
- Import from the standard library and third-party packages
- Read and write files using `with` statements and `pathlib`
- Handle exceptions with `try/except/finally`
- Design classes with `__init__`, properties, and dunder methods
- Understand how OOP maps directly to the scikit-learn API (`fit`, `transform`, `predict`)
- Write generator functions and use comprehensions for all four collection types
- Build and apply decorators and context managers

**Key Concepts Introduced**
Modules · OOP · classes · inheritance · generators · decorators · context managers

**Bridge:** The `SurveyAnalyser` class deliberately mirrors how scikit-learn estimators work —
this is the conceptual link between pure Python and ML frameworks.

**Project:** `SurveyLoader` + `SurveyAnalyser` classes operating on 500 SO 2025 rows

**Unlocks:** Chapter 3; the scikit-learn API in Chapter 6 will feel familiar

---

## Part 2 — Data Science Foundations

### Chapter 3 — NumPy and Pandas

| Property | Detail |
|----------|--------|
| **File** | `CH03_NumPy_and_Pandas.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 2 |
| **Background assumed** | Python OOP and file I/O |

**Learning Objectives**
- Create and manipulate NumPy arrays; understand shape, dtype, and broadcasting
- Apply universal functions (ufuncs) for vectorised computation
- Build and index Pandas Series and DataFrames with `.loc` and `.iloc`
- Clean real data: handle missing values, fix dtypes, detect and remove outliers
- Aggregate with `groupby`, reshape with `pivot_table`, combine with `merge`
- Produce `df_clean` — the cleaned dataset used in every subsequent chapter

**Key Concepts Introduced**
Arrays · broadcasting · vectorisation · DataFrames · `.loc`/`.iloc` · `groupby` · `merge`

**Mental Model:** NumPy arrays are the underlying storage for every ML framework tensor.
Understanding shape `(rows, cols)` here prevents debugging pain in Chapters 7–9.

**Project:** Full SO 2025 cleaning pipeline — 15,000 rows ready for ML

**Unlocks:** Chapters 4, 5, 6 (and indirectly everything after)

---

### Chapter 4 — Data Visualisation

| Property | Detail |
|----------|--------|
| **File** | `CH04_Data_Visualization.ipynb` |
| **Time** | 3–4 hours |
| **Prerequisites** | Chapter 3 |
| **Background assumed** | Pandas DataFrames |

**Learning Objectives**
- Use Matplotlib's Figure/Axes model for precise plot control
- Choose the right chart type: line, scatter, bar, histogram, box, violin
- Build multi-panel layouts with `plt.subplots`
- Apply Seaborn for statistical visualisations: `histplot`, `boxplot`, `heatmap`, `pairplot`
- Build interactive charts with Plotly Express
- Design charts for communication: titles, labels, colour, annotation

**Key Concepts Introduced**
Figure/Axes · `subplots` · Seaborn themes · FacetGrid · Plotly Express

**Project:** Full EDA suite on SO 2025 — salary distributions, language rankings, AI tool adoption

**Unlocks:** Chapter 5; visualisation skills used in every subsequent chapter

---

## Part 3 — Machine Learning and Artificial Intelligence

> Covers three data modalities in sequence: **tabular** (Ch 5–7), **text** (Ch 8),
> **images** (Ch 9). Chapter 10 audits all three for fairness.

### Chapter 5 — SciPy and Statistical Computing

| Property | Detail |
|----------|--------|
| **File** | `CH05_SciPy_Statistical_Computing.ipynb` |
| **Time** | 4–5 hours |
| **Prerequisites** | Chapter 4 |
| **Background assumed** | Basic statistics concepts (mean, variance) helpful but not required |

**Learning Objectives**
- Understand probability distributions and test for normality
- Run two-sample t-tests and Mann-Whitney U tests correctly
- Apply one-way ANOVA and chi-squared tests
- Compute and interpret effect sizes: Cohen's d, eta-squared, Cramér's V
- Fit custom curves and find roots with `scipy.optimize`
- Distinguish statistical significance from practical significance

**Key Concepts Introduced**
p-values · effect sizes · hypothesis testing · normality · confidence intervals · curve fitting

**Mental Model:** Statistical tests answer "is this difference real or noise?" — the same
question a learning curve answers in ML, just with different machinery.

**Project:** Is the Python salary premium statistically significant, and how large is the effect?

**Unlocks:** Chapter 6; statistical thinking informs every model evaluation decision

---

### Chapter 6 — Machine Learning with scikit-learn

| Property | Detail |
|----------|--------|
| **File** | `CH06_Machine_Learning_Sklearn.ipynb` |
| **Time** | 6–7 hours |
| **Prerequisites** | Chapter 5 |
| **Background assumed** | NumPy, Pandas, basic statistics |

**Learning Objectives**
- Explain the scikit-learn API contract: `fit`, `transform`, `predict`, `score`
- Build preprocessing pipelines that prevent data leakage
- Apply `ColumnTransformer` for mixed numeric/categorical data
- Train and evaluate regression models (Ridge, Lasso, Random Forest)
- Train and evaluate classification models (Logistic Regression, Gradient Boosting)
- Diagnose bias vs variance using learning curves
- Tune hyperparameters with `RandomizedSearchCV`
- Handle class imbalance with SMOTE and class weights
- Optimise decision thresholds using precision-recall curves
- Calibrate predicted probabilities with `CalibratedClassifierCV`
- Combine feature sets with `FeatureUnion`
- Cluster developers with KMeans; visualise with PCA

**Key Concepts Introduced**
Pipeline · ColumnTransformer · cross-validation · bias-variance tradeoff ·
data leakage · SMOTE · calibration · FeatureUnion · KMeans · PCA

**Mental Model:** A scikit-learn `Pipeline` is just a sequence of `fit/transform` steps
followed by a final `fit/predict` step — exactly the `SurveyAnalyser` pattern from Chapter 2.

**Project:** Salary regression · Python-usage classifier · developer clustering

**Unlocks:** Chapters 7, 10, 11 (all three depend on Chapter 6 concepts)

---

### Chapter 7 — Deep Learning with PyTorch

| Property | Detail |
|----------|--------|
| **File** | `CH07_Deep_Learning_PyTorch.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 6 |
| **Background assumed** | scikit-learn API; linear algebra basics helpful |
| **GPU** | **Required — enable T4 in Colab** |

**Learning Objectives**
- Create tensors and understand shape, dtype, and device placement
- Explain autograd: how PyTorch tracks operations for automatic differentiation
- Build neural networks with `nn.Module`, `nn.Linear`, `nn.BatchNorm1d`, `nn.Dropout`
- Write the five-step training loop: `zero_grad` → forward → loss → `backward` → `step`
- Use `model.train()` / `model.eval()` correctly
- Apply `ReduceLROnPlateau` and checkpoint best weights
- Save and load models with `torch.save` / `torch.load`
- Find optimal learning rates with the LR Range Test
- Accelerate training with mixed precision (`torch.cuda.amp`)
- Compile models for GPU efficiency with `torch.compile`

**Key Concepts Introduced**
Tensors · autograd · `nn.Module` · BatchNorm · Dropout · training loop ·
`BCEWithLogitsLoss` · `state_dict` · LR Range Test · AMP · `torch.compile`

**Mental Model:** The PyTorch training loop is the scikit-learn `fit()` method written out
explicitly — forward pass = predict, loss = score, backward = learn, step = update weights.

**Project:** Salary regression MLP · Python-usage binary classifier MLP

**Unlocks:** Chapters 8 and 9 (both use the PyTorch training loop)

---

### Chapter 8 — NLP and Transformers

| Property | Detail |
|----------|--------|
| **File** | `CH08_NLP_and_Transformers.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 7 |
| **Background assumed** | PyTorch training loop; basic text familiarity |
| **GPU** | **Recommended — T4 in Colab** |

**Learning Objectives**
- Build a classical NLP pipeline: clean → tokenise → TF-IDF → classify
- Explain subword tokenisation and why it handles unknown words
- Run zero-shot inference with HuggingFace `pipeline()`
- Fine-tune DistilBERT on a custom classification task
- Visualise attention weights to interpret model focus
- Build a full RAG pipeline: embed documents, index with FAISS, retrieve, generate
- Integrate API-based LLMs (OpenAI / Anthropic) in a provider-agnostic client
- Prompt for structured JSON output and parse responses safely

**Key Concepts Introduced**
TF-IDF · subword tokenisation · `[CLS]` token · attention · fine-tuning ·
sentence embeddings · FAISS · RAG · `LLMClient` · structured prompting

**Mental Model:** RAG separates *knowledge* (the document store, easily updated) from
*reasoning* (the LLM, fixed). Updating knowledge requires no retraining — just re-indexing.

**Project:** Developer role classifier (TF-IDF vs DistilBERT) · end-to-end RAG over SO 2025

**Unlocks:** Chapter 10 audits NLP models for bias; Chapter 11 deploys them

---

### Chapter 9 — Computer Vision with PyTorch

| Property | Detail |
|----------|--------|
| **File** | `CH09_Computer_Vision_PyTorch.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 7 |
| **Background assumed** | PyTorch training loop; Chapters 8 and 9 are independent |
| **GPU** | **Required — T4 in Colab** |

**Learning Objectives**
- Explain how convolutional layers detect spatial features via weight sharing
- Build a CNN from scratch using `nn.Conv2d`, `nn.MaxPool2d`, `nn.BatchNorm2d`
- Build an augmentation pipeline with `torchvision.transforms`
- Visualise what a CNN learns by inspecting feature maps with forward hooks
- Apply transfer learning: freeze a ResNet-18 backbone, replace the classification head
- Run object detection with a pre-trained Faster R-CNN and draw bounding boxes
- Run semantic segmentation with DeepLabV3 and visualise per-pixel label maps

**Key Concepts Introduced**
Convolution · weight sharing · receptive field · MaxPool · `torchvision` ·
data augmentation · transfer learning · feature maps · Faster R-CNN · DeepLabV3

**Mental Model:** CNN layers learn a hierarchy of features — edges → textures → shapes →
objects. Pre-trained models give you that hierarchy for free; you only teach the final categories.

**Project:** Custom CNN vs ResNet-18 on CIFAR-10 · bounding box detection · segmentation masks

**Unlocks:** Chapter 10 (ethics applies to CV models); Chapter 11 (deploy any of these)

---

### Chapter 10 — Ethics, Bias, and Responsible AI

| Property | Detail |
|----------|--------|
| **File** | `CH10_Ethics_Bias_Responsible_AI.ipynb` |
| **Time** | 4–5 hours |
| **Prerequisites** | Chapter 6 |
| **Background assumed** | scikit-learn models; basic statistics |
| **GPU** | Not required |

**Learning Objectives**
- Name and identify five sources of bias in ML systems
- Audit a dataset for demographic representation gaps
- Compute per-group fairness metrics: MAE by group, bias direction
- Explain and compare fairness criteria: demographic parity, equalised odds, calibration
- Generate global and local SHAP explanations for any sklearn model
- Apply sample reweighting as a bias mitigation technique
- Write a model card documenting intended use, limitations, and fairness metrics

**Key Concepts Introduced**
Representation bias · measurement bias · aggregation bias ·
demographic parity · equalised odds · SHAP · `shap.Explainer` · model cards

**Mental Model:** A model card is to an ML model what a nutrition label is to food —
it doesn't make a bad model good, but it gives users the information to make informed decisions.

**Project:** Full fairness audit of the Chapter 6 salary regression model

**Unlocks:** Chapter 11 (deploying responsibly requires the ethics vocabulary from Ch 10)

---

## Part 4 — Production and Deployment

### Chapter 11 — MLOps and Production ML

| Property | Detail |
|----------|--------|
| **File** | `CH12_MLOps_Production_ML.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 6 |
| **Background assumed** | scikit-learn models; basic REST API familiarity helpful |
| **GPU** | Not required |

**Learning Objectives**
- Explain the six-phase ML lifecycle and where MLOps fits
- Track experiments with MLflow: log parameters, metrics, and artefacts
- Register and stage models in the MLflow Model Registry
- Serve predictions as a REST API with FastAPI and Pydantic validation
- Write ML unit tests targeting data contracts and performance regression
- Detect data drift using PSI and the Kolmogorov-Smirnov test
- Generate a GitHub Actions CI/CD workflow that blocks degraded models from merging

**Key Concepts Introduced**
ML lifecycle · experiment tracking · model registry · `infer_signature` ·
FastAPI · Pydantic · PSI · KS test · GitHub Actions · performance regression tests

**Mental Model:** Training a model is ~10% of the work in a production ML system.
The other 90% is reproducibility, serving, monitoring, and keeping it working as the world changes.

**Project:** Full MLOps pipeline for the SO 2025 salary model — tracked, registered,
served, monitored, and CI-tested

**Unlocks:** Appendix G (Docker packages what you built here for deployment)

---

## Appendices

| Appendix | Title | File | Best read after |
|----------|-------|------|-----------------|
| A | Python Environment Setup | `APP_A_Environment_Setup.ipynb` | Chapter 0 |
| B | Keras 3 Companion | `APP_B_Keras3_Companion.ipynb` | Chapter 7 |
| C | Project Ideas & Further Reading | `APP_C_Projects_and_Further_Reading.ipynb` | Any chapter |
| D | Reinforcement Learning Foundations | `APP_D_Reinforcement_Learning.ipynb` | Chapter 7 |
| E | SQL for Data Scientists | `APP_E_SQL_for_Data_Scientists.ipynb` | Chapter 3 |
| F | Git & GitHub for ML | `APP_F_Git_GitHub_for_ML.ipynb` | Chapter 0 |
| G | Docker & Containerisation | `APP_G_Docker_Containerisation.ipynb` | Chapter 11 |

---

## Concept Dependency Graph

```
Ch 0 (Setup)
  └─► Ch 1 (Fundamentals)
        └─► Ch 2 (Intermediate Python)
              └─► Ch 3 (NumPy & Pandas)
                    └─► Ch 4 (Visualisation)
                          └─► Ch 5 (Statistics)
                                └─► Ch 6 (scikit-learn) ──────────────────────┐
                                      ├─► Ch 7 (PyTorch) ──────────────────┐  │
                                      │     ├─► Ch 8 (NLP)                 │  │
                                      │     └─► Ch 9 (Computer Vision)     │  │
                                      ├─► Ch 10 (Ethics) ◄─────────────────┘  │
                                      └─► Ch 12 (MLOps)  ◄────────────────────┘
```

---

## Learning Path Recommendations

| Your goal | Recommended path |
|-----------|-----------------|
| Complete beginner | Ch 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 |
| Python-literate, new to ML | Ch 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 |
| ML background, learn deep learning | Ch 7 → 8 → 9 |
| Want NLP / RAG | Ch 7 (prereq) → Ch 8 |
| Want computer vision | Ch 7 (prereq) → Ch 9 |
| Want production skills | Ch 6 (prereq) → Ch 11 → App G |
| Interview prep | Ch 6 + Ch 7 + App B (Keras) |
| Responsible AI focus | Ch 6 (prereq) → Ch 10 |

---

## Key Skills by Industry Role

| Role | Most relevant chapters |
|------|----------------------|
| Data Analyst | 0–4, App E (SQL) |
| ML Engineer | 6, 7, 10, 11, App F (Git), App G (Docker) |
| Data Scientist | 3–10 |
| NLP Engineer | 7, 8, App D |
| Computer Vision Engineer | 7, 9 |
| ML Platform / MLOps Engineer | 11, App F, App G |
| AI Researcher | 7, 8, 9, App D (RL) |

---

*Python for AI/ML — A Complete Learning Journey*
*Dataset: Stack Overflow 2025 Developer Survey*
*Platform: Google Colab (free) · Compatible with any Jupyter environment*

---

### Chapter 12 — Adversarial ML and Model Security

| Property | Detail |
|----------|--------|
| **File** | `CH11_Adversarial_ML_Security.ipynb` |
| **Time** | 5–6 hours |
| **Prerequisites** | Chapter 7 (PyTorch), Chapter 9 (CV), Chapter 8 (RAG) |
| **Background assumed** | PyTorch training loop; ML model deployment |
| **GPU** | **Recommended — T4 in Colab** |

**Learning Objectives**
- Map the ML attack surface: evasion, poisoning, extraction, and LLM-specific attacks
- Implement FGSM and PGD evasion attacks and measure accuracy degradation
- Apply adversarial training to harden a model; understand the robustness-accuracy tradeoff
- Simulate a label-flipping poisoning attack and detect it with kNN consistency scoring
- Execute a model extraction attack and apply rate limiting and output noise defences
- Apply a structured red team framework to an LLM RAG system

**Key Concepts Introduced**
FGSM · PGD · epsilon ball · adversarial training · label-flipping · model extraction ·
fidelity · prompt injection · indirect injection · red teaming · `RedTeamReport`

**Mental Model:** Security is a property of the *system*, not just the model weights.
An attacker has multiple entry points: the input, the training data, the API, and the
LLM's context. Each requires a different class of defence.

**Project:** Attack and defend the Chapter 9 ResNet-18 (evasion) and Chapter 8 RAG system (red team)

**Unlocks:** Appendix H (pipeline-level security)

---

### Appendix H — MLSecOps: Securing the ML Pipeline

| Appendix | Title | File | Best read after |
|----------|-------|------|-----------------|
| H | MLSecOps | `APP_H_MLSecOps.ipynb` | Ch 11 + Ch 12 |

**What's covered:** Supply chain risks · The pickle exploit and safetensors · FastAPI auth and rate limiting · `nbstripout` and `detect-secrets` · Distinguishing drift from adversarial probing · MLSecOps audit checklist

---

### Appendix I — Failure Modes and Troubleshooting

| Appendix | Title | File | Best read after |
|----------|-------|------|-----------------|
| I | Failure Modes | `APP_I_Failure_Modes.ipynb` | Any chapter where results seem wrong |

**What's covered:**
Silent data leakage (temporal, target, row) · Train/test contamination and the Pipeline fix ·
NaN propagation patterns by model type · Learning curves for fit diagnosis ·
Class imbalance and PR-AUC vs ROC-AUC · Gradient problems (vanishing, exploding, dead ReLUs) ·
Tokenisation gotchas (truncation, attention mask) · Training-serving skew ·
20-point diagnostic checklist + 'model performs worse than expected' decision tree

### Coding Exercises

All 13 chapters now include a **Coding Exercises** section with 3 exercises each:

| Tier | Symbol | Description |
|------|--------|-------------|
| Guided | 🔧 | Fill-in-the-blanks scaffold — practice the core pattern |
| Applied | 🔨 | Write from scratch on a new slice of the SO 2025 dataset |
| Extension | 🏗️ | Go beyond the chapter — combine techniques, investigate hypotheses |

**Total: 39 exercises** across 13 chapters. All use the SO 2025 dataset for consistency.
Collapsible hints and solutions in every exercise.

---

### Appendix J — Python Foundations: A Deep Dive

| Appendix | Title | File | Best read before |
|----------|-------|------|-----------------|
| J | Python Foundations | `APP_J_Python_Foundations.ipynb` | Chapter 0 (if new to programming) |

**What's covered:**
Variables, types, and the object model · Numbers and operators · String methods and f-strings ·
Control flow (if/elif/else) · Loops (for, while, break, continue) ·
Functions (parameters, defaults, *args, **kwargs, scope, lambda) ·
Lists in depth (methods, comprehensions, mutability) · Tuples and named tuples ·
Dictionaries in depth (iteration, comprehensions, common patterns) · Sets and set operations ·
Mutability and the reference model (the alias trap, shallow vs deep copy) ·
Error handling (try/except/else/finally, raising exceptions) ·
File I/O (reading, writing, CSV module) · Mini project (pure-Python salary analysis)

**Deliberately omits:** Type hints, dataclasses, generators, decorators, OOP —
those are covered in Chapters 1 and 2 once the foundations are solid.
