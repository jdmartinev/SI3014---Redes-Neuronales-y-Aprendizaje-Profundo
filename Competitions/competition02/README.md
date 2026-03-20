
---

## 📖 How to Approach This Lab

This lab is **not just coding**. It is designed to:

- Build **conceptual understanding**
- Develop **experimental skills**
- Produce a **self-contained technical report**

You must **follow this workflow strictly**:

---

## 1️⃣ Step 1 — Read the PDF FIRST

Before touching the notebooks, carefully read:

👉 `Laboratorio_2_DL.pdf`

This document defines:

- Objectives of the lab  
- Required models (M1, M2, M3)  
- Experimental design  
- Conceptual questions  
- Deliverables and deadlines  

⚠️ **Important:**  
Do NOT treat the notebook as instructions.  
The **PDF is the source of truth**.

---

## 2️⃣ Step 2 — Understand the Notebooks (Don’t Just Run Them)

Each notebook is a **guided implementation**, not a solution.

### 🔹 Part A — `TransferLearning_Workshop.ipynb`

Focus on:

- Dataset preprocessing (ImageNet normalization)
- Feature extraction vs fine-tuning
- Model adaptation:
  - VGG-16 (fixed + fine-tuned)
  - ResNet-18 (student-defined)
- Proper loss usage (CrossEntropy pitfalls)
- Experiment tracking (TensorBoard)

---

### 🔹 Part B — `UNet_Workshop.ipynb`

Focus on:

- Difference between **classification vs segmentation**
- U-Net architecture:
  - Encoder / Decoder
  - Skip connections
- Loss functions:
  - BCE vs Dice vs Combined
- Transfer learning:
  - ResNet encoder inside U-Net
- Evaluation metrics:
  - IoU
  - Dice Score

---

## 3️⃣ Step 3 — Implement, Don’t Copy

You are expected to:

- Complete missing parts of the code
- Make design decisions (especially in M3 models)
- Run experiments and compare results

Key expectation:

> You understand **why** each step is done, not just **how**

---

## 4️⃣ Step 4 — Answer Conceptual Questions

Each section includes **research questions** such as:

- When NOT to use transfer learning?
- Why Dice works better than CrossEntropy in segmentation?
- What problem do residual connections solve?
- Why freezing layers helps?

👉 These must be answered **inside the notebook (Markdown cells)**.

---

## 5️⃣ Step 5 — Track and Compare Experiments

You must:

- Log training curves (loss, metrics)
- Use **TensorBoard**
- Compare:
  - Frozen vs fine-tuned models
  - Architectures (VGG vs ResNet)
  - Baseline vs pretrained encoders

---

## 6️⃣ Step 6 — Deliverable Requirements

You must submit:

### 📌 One notebook per part:

- `grupoX_LAB#2A.ipynb`
- `grupoX_LAB#2B.ipynb`

Each notebook must be:

- ✅ Fully executed  
- ✅ Self-contained  
- ✅ With answers in Markdown  
- ✅ With visible results (plots, metrics)

> Someone should understand your solution **without running it**

---

## ⏰ Deadlines

- **Part A:** March 23, 2026 – 11:59 PM  
- **Part B:** March 27, 2026 – 11:59 PM  

---

## 📧 Submission Format

Send via **email only** with subject:

- `Laboratorio #2 - Grupo # - Parte A`
- `Laboratorio #2 - Grupo # - Parte B`

---

## ⚠️ Common Mistakes (Avoid These)

- ❌ Using `LogSoftmax` + `CrossEntropyLoss` together  
- ❌ Not freezing layers during transfer learning  
- ❌ Misaligned image/mask transforms in segmentation  
- ❌ Not normalizing inputs for pretrained models  
- ❌ Submitting notebooks without outputs  

---

## 🎯 What You Should Learn

By the end of this lab, you should be able to:

- Apply **transfer learning properly**
- Understand **feature reuse vs fine-tuning**
- Build **segmentation models from scratch**
- Use **pretrained encoders in new tasks**
- Evaluate models using **appropriate metrics**
