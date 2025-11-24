# Experiment Summary

## Overview

This document summarizes experiments analyzing LLM behavior in a hierarchical action sequence learning task (star-making paradigm). The task involves learning 4-key action sequences during a learning phase and applying them during a transfer phase.

---

## 1. Learning Phase Analysis (`learning_phase.py`)

### Hypothesis
During learning, participants will progressively discover and preferentially use valid 2-key action sequences (chunking) over invalid sequences, and this effect will increase over trials.

### Key Results

| Condition | Validity Effect (p) | Trial Effect (p) | Validity x Trial Interaction (p) |
|-----------|---------------------|------------------|----------------------------------|
| Transfer Low / No Item | p < 0.001 (negative) | p = 0.35 (ns) | p < 0.001 (positive) |
| Transfer Low / Item | p < 0.001 (positive) | p < 0.001 (negative) | p < 0.001 (positive) |
| Transfer High / No Item | p < 0.001 (negative) | p = 0.07 (ns) | p = 0.04 (positive) |
| Transfer High / Item | p < 0.001 (positive) | p < 0.001 (negative) | p < 0.001 (positive) |

### Interpretation
- **Supported**: The significant validity x trial interaction across all conditions shows that valid 2-key sequences increase relative to invalid sequences over trials
- The **item condition** shows stronger chunking effects (positive validity main effect, meaning more valid than invalid sequences)
- The **no-item condition** shows initial preference for invalid sequences but learns to correct this over trials

---

## 2. Transfer Phase Analysis (`transfer_phase.py`)

### Experiment 2.1: 2-Key Sequence Transfer Times

**Hypothesis**: Valid 2-key sequences discovered in one slot will be transferred to the other slot faster than invalid sequences.

**Results (Item condition)**:
- Block-centered effect: B = 0.44, z = 3.25, p = 0.001
- Transfer time difference (valid - invalid) increases over blocks

**Results (No-Item condition)**:
- Block-centered effect: B = -0.04, z = -0.56, p = 0.57 (ns)

**Interpretation**: With item feedback, participants show faster transfer of valid 2-key chunks. Without items, no differential transfer is observed.

### Experiment 2.2: Transfer Accuracy Comparison (High vs Low)

**Hypothesis**: Transfer accuracy will differ between high and low transfer conditions.

**Results (Item condition)**:
- Transfer type effect: B = 0.028, z = 10.2, p < 0.001
- Interaction (type x trial): B = 0.0014, z = 6.14, p < 0.001
- High transfer shows better accuracy that improves over trials

**Results (No-Item condition)**:
- Transfer type effect: B = -0.05, z = -16.8, p < 0.001
- Low transfer actually shows better accuracy than high transfer

**Interpretation**: Item feedback enables better transfer to high-similarity contexts; without items, low transfer (more different) is paradoxically easier.

---

## 3. Item Stimulus Effect (`item_stimulus.py`)

### Experiment 3.1: Effect on 2-Key Sequence Chunking

**Hypothesis**: Item feedback during learning improves chunking of 2-key sequences and facilitates their reuse during transfer.

**Learning Phase Model (2-Key Sequences)**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Validity | -0.081 | -5.72 | < 0.001 |
| Trial | -0.009 | -6.87 | < 0.001 |
| Item Feedback | 0.042 | 2.92 | 0.003 |
| Validity x Item | 0.245 | 12.14 | < 0.001 |
| Validity x Trial x Item | 0.027 | 10.86 | < 0.001 |

**Transfer Phase Model (2-Key Sequences)**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Validity | 0.246 | 18.08 | < 0.001 |
| Item Feedback | -0.200 | -14.68 | < 0.001 |
| Validity x Item | 0.660 | 34.31 | < 0.001 |
| Validity x Trial x Item x Transfer | 0.018 | 5.48 | < 0.001 |

**Interpretation**:
- **Strongly Supported**: Item feedback dramatically increases valid sequence usage
- The 3-way interaction (validity x trial x item) shows item feedback accelerates learning of valid chunks
- During transfer, item-trained models show much stronger preference for valid sequences (validity x item B = 0.66)
- The 4-way interaction shows this effect is modulated by transfer type

### Experiment 3.2: Effect on Learning Accuracy (First 40 Trials)

**Hypothesis**: Item feedback improves model performance (accuracy) during learning.

**Learning Accuracy Model**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Item Feedback | 0.056 | 14.82 | < 0.001 |
| Trial | -0.002 | -10.66 | < 0.001 |
| Item x Trial | 0.002 | 6.87 | < 0.001 |

**Interpretation**:
- **Supported**: Item feedback significantly improves learning accuracy (B = 0.056, ~5.6% higher)
- Accuracy decreases over trials as task difficulty increases
- The interaction shows item feedback helps maintain accuracy over trials (B = 0.002)

### Experiment 3.3: Effect on Transfer Accuracy (First 40 Trials)

**Hypothesis**: Item feedback improves model performance (accuracy) during transfer.

**Transfer Accuracy Model**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Item Feedback | -0.043 | -14.51 | < 0.001 |
| Trial | -0.0002 | -1.35 | 0.178 (ns) |
| Transfer Type | -0.050 | -16.74 | < 0.001 |
| Item x Trial | 0.0001 | 0.54 | 0.586 (ns) |
| Item x Transfer Type | 0.078 | 18.34 | < 0.001 |
| Item x Trial x Transfer Type | 0.001 | 5.54 | < 0.001 |

**Interpretation**:
- **Partially Supported**: Item feedback shows a complex effect on transfer accuracy
- Main effect of item feedback is negative (B = -0.043), but this is qualified by interactions
- Critical interaction: Item x Transfer Type (B = 0.078) shows item feedback improves high transfer accuracy relative to low transfer
- The 3-way interaction confirms item feedback benefits are stronger in high transfer and improve over trials

---

## 4. Sequence Transfer and Reuse (`sequence_item_transfer_usage.py`)

### Experiment 4.1: 4-Key Sequence Reuse

**Hypothesis**: Previously rewarded 4-key sequences from learning will be reused during transfer.

**Results (Item condition)**:
| Condition | Intercept | Trial Effect | p |
|-----------|-----------|--------------|---|
| Transfer High | 0.234 | -0.0009 | 0.016 |
| Transfer Low | 0.215 | 0.00006 | 0.87 (ns) |

**Results (No-Item condition)**:
| Condition | Intercept | Trial Effect | p |
|-----------|-----------|--------------|---|
| Transfer High | 0.00 | -0.003 | < 0.001 |
| Transfer Low | 0.126 | -0.005 | < 0.001 |

**Interpretation**:
- With items: ~22% of transfer sequences are OLD (from learning), stable over trials
- Without items: OLD sequence reuse is near zero or rapidly declines
- Item feedback creates persistent memory representations

### Experiment 4.2: 2-Key Valid vs Invalid Usage

**Results (Item condition)**:
- Seq type indicator (valid vs invalid): B = 0.91, z = 79.1, p < 0.001
- Strong preference for valid sequences (~91% higher)

**Results (No-Item condition)**:
- Seq type indicator: B = 0.17-0.19, z = 13-14, p < 0.001
- Weaker but significant preference for valid sequences

**Interpretation**: Item feedback creates much stronger 2-key chunking (5x larger effect size).

---

## 5. Success vs Failure Case Analysis (`success_failure_cases.py`)

### Experiment 5.1: Learning Phase Success Predicts Exploration

**Hypothesis**: High-success participants show different exploration patterns (more valid 2-key usage) than low-success participants.

**Results (High transfer condition)**:
- Group x Trial interaction: B = 0.014, z = 6.08, p < 0.001
- High-success participants increase valid sequence usage faster over trials

**Results (Low transfer condition)**:
- Group x Trial interaction: B = 0.005, z = 2.40, p = 0.016
- Similar pattern but weaker effect

**Interpretation**: Successful learners show accelerated discovery of valid action chunks.

### Experiment 5.2: Transfer Success Predicts Learning Exploration

**Hypothesis**: Participants who succeed in transfer show different learning phase exploration.

**Results (High transfer)**:
- Group x Trial interaction: B = 0.006, z = 3.29, p < 0.001
- Transfer success associated with faster learning of valid sequences

**Results (Low transfer)**:
- Group x Trial interaction: B = 0.003, z = 1.56, p = 0.12 (ns)
- No significant relationship in low transfer condition

**Interpretation**: Transfer success is predicted by learning phase exploration in high-similarity transfer contexts.

### Experiment 5.3: Learning-Transfer Correlation

**Hypothesis**: Learning success correlates with transfer success.

**Results**:
| Condition | Pearson r | p |
|-----------|-----------|---|
| High | -0.015 | 0.87 (ns) |
| Low | -0.020 | 0.83 (ns) |

**Interpretation**: **Not Supported** - No significant correlation between learning and transfer success. This suggests transfer requires additional capabilities beyond learning performance.

---

## Summary Table

| Experiment | Hypothesis | Support |
|------------|------------|---------|
| Learning Phase Chunking | Valid sequences increase over trials | Supported |
| Transfer Time Difference | Valid sequences transfer faster | Partially Supported (Item only) |
| High vs Low Transfer | High transfer shows better accuracy | Mixed (depends on item condition) |
| Item Feedback on Chunking | Items improve 2-key chunking | Strongly Supported |
| Item Feedback on Learning Accuracy | Items improve learning performance | Supported |
| Item Feedback on Transfer Accuracy | Items improve transfer performance | Partially Supported (High transfer) |
| 4-Key Sequence Reuse | OLD sequences reused in transfer | Supported (Item only) |
| 2-Key Preference | Valid > Invalid during transfer | Strongly Supported |
| Success Predicts Exploration | High-success shows better chunking | Supported |
| Learning-Transfer Correlation | Learning predicts transfer | Not Supported |

---

## Key Conclusions

1. **Item feedback is critical**: It enables formation of robust action chunks that persist into transfer
2. **Chunking develops over learning**: Valid 2-key sequences increase relative to invalid ones across all conditions
3. **Transfer is context-dependent**: High-similarity transfer benefits from item-based learning; low-similarity transfer may not
4. **Learning and transfer are dissociated**: Success in learning does not predict transfer success, suggesting transfer requires additional mechanisms
