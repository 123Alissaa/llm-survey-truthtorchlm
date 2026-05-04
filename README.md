# llm-survey-truthtorchlm

# Survey: Evaluating Truthfulness Prediction Methods for Large Language Models via TruthTorchLM

**Authors:** Alissa Josy · Renu Chikkam  
**University of South Florida**

## Overview

This repository contains the code and results for our survey project 
evaluating truthfulness prediction methods implemented in TruthTorchLM, 
an open-source Python library that unifies 30+ truth methods under a 
single interface. We conduct three experiments:

- **Experiment 1** : Short-form QA reproduction on TriviaQA 
  and GSM8K using LLaMA-3-8B and GPT-4o-mini
- **Experiment 2** : Long-form evaluation on FactScore-Bio 
  using LLaMA-3-8B
- **Experiment 3** : Cross-domain generalization analysis 
  between TriviaQA and GSM8K

## Repository Structure
llm-survey-truthtorchlm/
├── Renu_Alissa_TruthTorchLM_Survey.ipynb   # Experiments 1 and 3
├── truthtorch-exp2.ipynb                    # Experiment 2
├── experiment2_raw_results(1).json             # Raw results Experiment 2
├── evaluation_metrics.csv                   # Raw results Experiment 1
├── evaluation_metrics2(1).csv              # Raw results Experiment 1
├── evaluation_metrics3.csv                  # Raw results Experiment 1
├── evaluation_metrics4.csv                  # Raw results Experiment 1
├── experiment3results.png                   # Experiment 3 visualization
└── README.md                             

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o-mini access)
- HuggingFace account with LLaMA-3-8B-Instruct access approved

### Additional for Experiment 2 only
- Kaggle account with GPU T4 x2 enabled
- Serper API key (free at serper.dev)

## How to Run Experiment 1 and 3

### Setup
1. Open `Renu_Alissa_TruthTorchLM_Survey.ipynb` in Google Colab
2. Select **A100 GPU** under Runtime → Change runtime type
3. Set your API keys at the top of the notebook:
```python
os.environ["OPENAI_API_KEY"] = "your key here"
```

### What it does
- Loads LLaMA-3-8B locally
- Calibrates truth methods using Isotonic Regression on 200 samples
- Evaluates on 200 test samples each for TriviaQA and GSM8K
- Uses GPT-4o-mini as model judge for correctness labeling
- Computes AUROC and PRR for LARS, SAR, VerbalizedConfidence, 
  and MultiLLMCollab
- Experiment 3 analyzes cross-domain AUROC degradation between 
  TriviaQA and GSM8K

### Expected Runtime
Approximately 10 hours on Google Colab A100 High-RAM for 200 samples.

---

## How to Run Experiment 2

### Setup
1. Upload `truthtorch-exp2.ipynb` to Kaggle
2. Enable **GPU T4 x2** under Settings → Accelerator
3. Add the following Kaggle Secrets (lock icon in sidebar):
   - `OPENAI_API_KEY` — your OpenAI API key
   - `SERPER_API_KEY` — your Serper API key
   - `HF_TOKEN` — your HuggingFace token

### Request LLaMA-3-8B Access
Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 
and request access. Approval usually takes a few hours.

### Run the Notebook
Run all cells in order from top to bottom. The notebook will:
1. Install all dependencies
2. Load API keys from Kaggle Secrets
3. Load LLaMA-3-8B on GPU
4. Define 6 truth methods (LARS, SAR, PTrue, Confidence, 
   VerbalizedConfidence, TokenSAR)
5. Set up the long-form pipeline (decomposition + SAFE)
6. Load 20 FactScore-Bio biography questions
7. Run the full evaluation pipeline
8. Save results to `experiment2_raw_results.json`
9. Generate tables and visualizations

### Expected Runtime
Approximately 1 hour 37 minutes on Kaggle T4 x2 for 20 questions.

## Key Dependencies

### Experiments 1 and 3
TruthTorchLM
transformers
torch
litellm
datasets

### Experiment 2

transformers==4.40.0
TruthTorchLM
litellm
accelerate
google-search-results
datasets
minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main

**Note:** transformers must be pinned to 4.40.0 for Experiment 2 
due to a tokenizer compatibility issue with newer versions.

## Results Summary

### Experiment 1 — LLaMA-3-8B (TriviaQA / GSM8K)

| Method | TriviaQA AUROC | GSM8K AUROC |
|---|---|---|
| LARS | 0.847 | 0.873 |
| SAR | 0.785 | 0.801 |
| VerbalizedConfidence | 0.739 | 0.546 |
| MultiLLMCollab | 0.662 | 0.690 |

### Experiment 2 — LLaMA-3-8B (FactScore-Bio)

| Method | AUROC | PRR |
|---|---|---|
| LARS | 0.698 | 0.504 |
| PTrue | 0.661 | 0.189 |
| SAR | 0.597 | 0.050 |
| VerbalizedConfidence | 0.542 | 0.184 |
| TokenSAR | 0.462 | 0.044 |
| Confidence | 0.426 | -0.082 |

## References

- TruthTorchLM: https://github.com/Ybakman/TruthTorchLM
- Yaldiz et al. (2025): TruthTorchLM: A Comprehensive Library for 
  Predicting Truthfulness in LLM Outputs. EMNLP 2025.
