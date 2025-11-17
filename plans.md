# Judica

## 1. Goal

Train **Judica**, an open-source Qwen-based model that:

* Takes as input:

  * a petition / AAO case summary (EB-1A / NIW / O-1)
* Outputs:

  * structured judgment (per-criterion: met / not_met / unclear)
  * overall strength (weak / borderline / strong)
  * short rationale in AAO-style language

Training will be done in **two stages**:

1. **SFT (Supervised Fine-Tuning)** – teach format, rubric, and rough judgment.
2. **GRPO (Reinforcement Fine-Tuning)** – sharpen strictness and ranking using quiz-style rewards.

At this point we assume **no data or checkpoints exist yet**; this is the plan for when they do.

---

## 2. Phase 1 – SFT (Supervised Fine-Tuning)

**Objective:**
Make Qwen reliably behave like a structured EB-1A/NIW judge.

### 2.1 Data (future)

Each training example will be:

* **Input:**

  * `petition_or_decision_text`: main body of the petition or AAO decision summary.
* **Target output (what we want the model to learn):**

  * JSON-like structure, e.g.:

    ```json
    {
      "category": "EB1A",
      "final_decision": "weak | borderline | strong",
      "criteria": {
        "awards": "met | not_met | unclear",
        "original_contributions": "met | not_met | unclear",
        "leading_role": "met | not_met | unclear",
        "high_salary": "met | not_met | unclear"
      },
      "analysis": "2–5 sentences explaining why."
    }
    ```

Labels will be derived later from AAO decisions + curated good examples.

### 2.2 Training setup

* **Base model:** e.g., `Qwen2.5-7B/14B-Instruct` or similar.
* **Method:** LoRA/QLoRA SFT using a framework like LLaMA-Factory or Axolotl.
* **Loss:** standard cross-entropy on the target text (the JSON output).

### 2.3 What SFT should achieve

After SFT, Judica should:

* Always output valid, parseable JSON in the expected schema.
* Produce criterion labels that roughly match human/AAO labels.
* Give coherent, specific rationales (not generic fluff).
* Be stable enough to use for evaluation and for RL data generation.

This gives you a **usable v0 judge** and a solid starting checkpoint for GRPO.

---

## 3. Phase 2 – GRPO (Reinforcement Fine-Tuning with Quiz Reward)

**Objective:**
Push Judica from “copies labels reasonably” to “makes strict, AAO-consistent judgments and rankings.”

### 3.1 Preference / reward signal (future)

For each AAO case, we will build:

* **Quiz ground truth** from the decision, e.g.:

  ```json
  {
    "final_decision": "dismissed",
    "criteria": {
      "awards": "not_met",
      "original_contributions": "not_met",
      "leading_role": "met",
      "high_salary": "met"
    }
  }
  ```

* **Reward:** compare Judica’s predicted labels (from its JSON output) with these answers:

  * Higher reward if more fields match AAO.
  * Penalty if JSON format is broken or fields are missing.

Optionally add **preference pairs**, e.g.:

* pre-RFE vs post-RFE versions of the same case,
* or denial vs “fixed” synthetic version → good should score higher than bad.

### 3.2 GRPO loop (conceptual)

For each training step (once we have data + SFT checkpoint):

1. Pick a batch of petitions / AAO cases.
2. For each case, **sample N generations** from Judica (e.g., 8 outputs with some temperature).
3. For each output:

   * Parse its JSON.
   * Compute **quiz reward** = how many fields match AAO ground truth.
   * Add **format reward** for valid JSON and required fields.
4. Within each group of N outputs for the same case, compute **group-relative advantages** (reward minus group mean).
5. Use GRPO (policy vs frozen reference model) to update Judica **toward higher-reward generations**.

### 3.3 What GRPO should improve

Compared to pure SFT, GRPO should:

* Make Judica **stricter and more AAO-like** (less over-approval).
* Improve **ranking**: good vs bad versions of the same profile.
* Stabilize labels under sampling (less variance between runs).
* Maintain output format while optimizing judgment quality.

---

## 4. Summary

* **SFT:**

  * Trains Judica to speak in the right **format** and follow the **rubric**.
  * Uses AAO decisions + curated good petitions as supervised targets.
  * Produces a functional in-domain judge.

* **GRPO:**

  * Uses AAO-derived **quiz answers** + preference pairs as rewards.
  * Optimizes Judica’s behavior **relatively** within sampled groups.
  * Sharpens its strictness, consistency, and ranking ability.

You don’t need any data or checkpoints to start with this plan; this is the scaffold you’ll fill in as you collect AAO decisions and your own labeled examples.
