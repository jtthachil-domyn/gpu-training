# 01 — ELI5 Intro (Explain-to-lead / five-year-old level)

Training a model from scratch is like teaching a robot to read lots of books so it can answer questions.

- Colab/Kaggle = one small desk where one person can read a few books slowly.
- A cluster = a whole classroom where many people read books together, splitting the pile and finishing faster.

We want to teach our team *how to work in that classroom* even though we currently only have access to a “desk.”

The team will learn:

- how to split work (data parallelism),
- how to keep everyone synchronized (gradient reduction),
- how to save the robot’s progress (checkpointing),
- how to make sure the robot is learning correctly (evaluation, metrics).

We will run:

- Short lessons
- Practical labs
- End-to-end training runs (tiny models)
- Multi-process simulations (no GPU cluster needed yet)

Once everyone understands this, we will move to real GPU hardware and run the same workflows at full scale.

---

## Why this ELI5 approach matters

Leads and non-technical stakeholders need a mental model before discussing:

- GPU budgets
- Cluster requirements
- Model scaling limits
- Training time estimates
- ROI from distributed model training

This simple explanation is consistent with the technical content to follow.
