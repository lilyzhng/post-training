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


# SFT and GRPO Loss

## 1. SFT loss (Judica as a supervised model)

You’re just doing standard next-token cross-entropy on your **target judgment output** (the JSON with scores + rationale).

Let:

* $x$ = input text (petition / AAO summary)
* $y = (y_1, \dots, y_T)$ = target output tokens (your labeled Judica JSON + rationale)
* $\pi_\theta$ = your Qwen model with parameters $\theta$

**SFT loss:**

$$
\mathcal{L}_{\text{SFT}}(\theta)
= - \mathbb{E}_{(x,y)\sim \mathcal{D}} \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})
$$

In practice:

* You tokenize the **system + user** as input and **assistant** as labels.
* You mask loss on input tokens; only compute CE on assistant tokens.

You can optionally add:

* **Format penalty**: extra CE on special tokens/JSON brackets if you really care about structure, but usually plain CE is enough if targets are clean.

---

## 2. GRPO loss (quiz-based RL over groups)

Now you have:

* A **frozen reference model** $\pi_{\text{ref}}$ (copy of Judica SFT at start of RL).
* A **policy** $\pi_\theta$ you're updating.
* For each case $k$, you sample a **group** of N generations $\{y^{(k,1)}, \dots, y^{(k,N)}\}$.
* For each generation, you compute a **scalar reward** $R^{(k,i)}$ (quiz accuracy + format bonus).

### 2.1 Group-relative advantage

For each group $k$:

$$
\bar{R}^{(k)} = \frac{1}{N} \sum_{i=1}^N R^{(k,i)}
$$

$$
A^{(k,i)} = R^{(k,i)} - \bar{R}^{(k)}
$$

So within the group, "better than average" generations have **positive** advantage, worse ones **negative**.

### 2.2 Log-probabilities under policy & reference

For each sample:

* Full-sequence logprob under policy:

$$
\log \pi_\theta(y^{(k,i)} \mid x^{(k)}) = \sum_t \log \pi_\theta(y^{(k,i)}_t \mid x^{(k)}, y^{(k,i)}_{<t})
$$

* Same under reference:

$$
\log \pi_{\text{ref}}(y^{(k,i)} \mid x^{(k)})
$$

Define **log-ratio** and ratio:

$$
\Delta^{(k,i)} = \log \pi_\theta - \log \pi_{\text{ref}}
$$

$$
r^{(k,i)} = \exp(\Delta^{(k,i)}) = \frac{\pi_\theta}{\pi_{\text{ref}}}
$$

### 2.3 GRPO loss (PPO-style, but group-relative)

The simplest version (no clipping) is:

$$
\mathcal{L}_{\text{GRPO}}(\theta)
= - \mathbb{E}_{k,i} \Big[ A^{(k,i)} \cdot r^{(k,i)} \Big]
$$

Intuition:

* If $A^{(k,i)} > 0$: we want $r^{(k,i)}$ to grow (increase log-prob of that sample).
* If $A^{(k,i)} < 0$: we want $r^{(k,i)}$ to shrink (decrease log-prob).

In practice you often also:

* Add **clipping** like PPO, e.g.:

$$
\mathcal{L}_{\text{GRPO}} = - \mathbb{E}\Big[ \min(r A, \operatorname{clip}(r, 1-\epsilon, 1+\epsilon) A) \Big]
$$

* Add a small **KL penalty** to keep $\pi_\theta$ close to $\pi_{\text{ref}}$:

$$
\mathcal{L}_{\text{KL}} = \beta \, \mathbb{E}\big[ \text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \big]
$$

And total RL loss:

$$
\mathcal{L}_{\text{RL}} = \mathcal{L}_{\text{GRPO}} + \mathcal{L}_{\text{KL}}
$$

---

## 3. How they combine for Judica

Typical recipe for you:

1. **Stage 1 – SFT only**

$$
\mathcal{L} = \mathcal{L}_{\text{SFT}}
$$

Train until Judica outputs clean JSON + sane judgments.

2. **Stage 2 – RL (GRPO) on top**

Either:

* Pure RL:

$$
\mathcal{L} = \mathcal{L}_{\text{RL}}
$$

or

* Mixed objective (helps stability on small data):

$$
\mathcal{L} = \lambda_{\text{SFT}} \mathcal{L}_{\text{SFT}} + \lambda_{\text{RL}} \mathcal{L}_{\text{RL}}
$$

with $\lambda_{\text{SFT}} \ll \lambda_{\text{RL}}$ (e.g., 0.1 vs 1.0).

You can think of:

* **SFT loss** → "speak like a judge, follow the rubric, output JSON."
* **GRPO loss** → "be *more right* according to AAO quizzes and preference pairs than your own SFT baseline."

If you want, I can sketch the exact quiz-based reward term $R$ in equation form next.

