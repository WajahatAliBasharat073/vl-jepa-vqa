# VL-JEPA for Visual Question Answering

## Overview
This repository documents my MS thesis research at NUST Pakistan on replacing 
CLIP-style contrastive embeddings with VL-JEPA representations for fine-grained 
visual grounding and reasoning in Visual Question Answering (VQA).

## Research Question
Can predictive representation learning (JEPA) improve visual grounding and 
reasoning in VQA compared to contrastive learning (CLIP)?

## Motivation
Current VQA systems rely on CLIP-style embeddings trained with contrastive 
objectives. These embeddings match image-text pairs globally but fail to capture 
fine-grained visual semantics — a critical limitation for tasks requiring precise 
visual grounding and detailed reasoning.

VL-JEPA learns by predicting abstract representations of masked regions rather 
than reconstructing raw inputs. This should produce richer, more detailed internal 
representations of visual content — exactly what fine-grained VQA requires.

## Approach
- Replace CLIP encoder with VL-JEPA representations in a VQA pipeline
- Evaluate on standard VQA benchmarks: GQA, VQAv2, TallyQA
- Compare representation quality for fine-grained visual grounding tasks
- Analyze where predictive representations outperform contrastive embeddings

## Background
This direction builds on the JEPA architecture family:
- **I-JEPA** (2023) — self-supervised image representations
- **V-JEPA** (2024) — self-supervised video representations  
- **VL-JEPA** (2025) — vision-language representation learning

Key insight from VL-JEPA: models trained to predict meaning rather than 
reconstruct surface details learn better representations. This thesis tests 
whether that advantage holds specifically for fine-grained visual grounding in VQA.

## Status
Active MS thesis research — NUST Pakistan, 2024–2026.

## References
- VL-JEPA (2025): https://arxiv.org/abs/2505.13954
- MAViL (2023): https://arxiv.org/abs/2212.08071
- I-JEPA (2023): https://arxiv.org/abs/2301.08243
