<!-- ---
title: DataClean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
--- -->

# DataCleanEnv — Reinforcement Learning for Autonomous Data Quality

[![Phase 1](https://img.shields.io/badge/Phase%201-Passed-brightgreen)]()
[![Phase 2](https://img.shields.io/badge/Phase%202-Passed-brightgreen)]()
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/huggingface/openenv)
[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/debashish1/dataclean-env)
[![Tasks](https://img.shields.io/badge/Tasks-3%20(Easy%20→%20Hard)-orange)]()

---

## The Problem

Data scientists spend **60–80% of their time cleaning data** — not building models.
This is universally acknowledged as the most painful, repetitive, and 
consequential task in the field.

**The cost is enormous:**
- A single mishandled null value in a financial dataset can cause 
  incorrect regulatory reporting
- A missed duplicate in a customer database leads to double-billing
- A wrong date format in a medical dataset can corrupt survival analysis

**No benchmark exists to train or evaluate agents on this task.**

Rule-based cleaners fail because the right operation depends on:
- Context (null rate + outlier presence + column type simultaneously)
- Purpose (dashboard prep requires imputation; ML pipelines require dropping)
- Order (outliers must be removed BEFORE imputation, not after)
- Interaction effects (cleaning column A changes what is optimal for column B)

DataCleanEnv is the first OpenEnv environment that frames data cleaning
as a sequential decision problem and provides the training ground for
agents that learn to clean purposefully, efficiently, and correctly.

---

## Why Reinforcement Learning — The Proof

Three properties make data cleaning a natural RL problem and make every
other approach provably insufficient:

**Rule-based systems fail** because the same null rate (say 12%) requires
median imputation when outliers are present, mean imputation when the 
distribution is symmetric, and row dropping when the column is a required
key field. No rule captures this — it requires context-sensitive judgment.

**Supervised learning fails** because no labeled dataset of correct 
cleaning sequences exists anywhere. Data scientists do not record their
decision-making steps. The label space is exponential.

**RL succeeds** because:
- Cleaning is a sequential decision process with order-dependent consequences
- The reward signal (comparison to ground truth) is available at every step
- The agent discovers context-sensitive policies through consequence-learning
- Trained agents generalize — a policy trained on 10,000 synthetic datasets
  transfers to real dirty data it has never seen

---

## Live Demo

See the environment in action without any setup: