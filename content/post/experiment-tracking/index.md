---
title: Lightweight experiment tracking workflow

summary: |-
    Taming chaos around computer experiments in pure Python

tags:
- machine learning
- research

date: "2024-04-23T10:18:00Z"

draft: true

# Optional external URL for project (replaces project detail page).
external_link: ""

# image:
#   caption: Photo by Michael Daniel 
#   focal_point: Smart
# url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
---

If you ever tried to do computer experiments, you know how hard it is to tame the chaos around them. With classical simulations (e.g. you solve a differential equation), your experiment usually consists of the code and the parameters. You run the same code with different parameters several times and record the results into a bunch of csv files. Then you modify your code, add some parameters, produce more data. Then you want to reproduce some previous results, but you lost the version of the code you used before. Fortunately, you use git, so it is not lost permanently. You just spend an hour to find the correct revision in the commit history. And then you realise that you want not only rerun the old experiment, but also extract some specific data from it. To do so, you have to use a feature that is implemented only in your most recent code, so you have to backport it to the previous version. You spend the rest of the day doing that and happily start the calculations just before going home. Next morning you turn on your computer and find that your recent calculations completely broke the csv files with the results because of the different set of columns!

With machine learning experiments, it becomes even more complicated, because now you have to track also your input data. There are specialized tools like [DVC](https://dvc.org) and [MLFlow](https://mlflow.org/) that are designed exactly for this purpose. They have a lot of nice features and are definitely worth considering, but often you don't want to learn and manage a new multifunctional tool just to make your workflow a bit less chaotic. And, in fact, you probably don't need to!

In this post I describe a simple framework that I use to track my numeric experiments (no matter, classical or ML-related) just with the usual Python tools.

## Requirements

1. **Reproducibility.** For each experiment result, it should be easy to reproduce it, ideally, by just running one command.
2. **Flexiblity and opennes.** I want to be able to modify the code I use to run experiments, introduce new parameters, add new features, etc. Taking into account previous requirement, these changes have to be backward compatible, i.e. do not break my previous experiments.
3. **Parallelization.** Several experiments can be run in parallel (i.e. on different computational nodes using Slurm), this should not break anything.
4. **Comparability.** I want to obtain one dataframe that collects the results of all (or some) experiments to be able to compare them across the experiments.
