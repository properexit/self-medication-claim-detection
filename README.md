# Claim Detection and Span Extraction in Reddit Health Discourse

This project implements a modular NLP pipeline for detecting and extracting self-medication claims from Reddit health discussions.

1.  Binary claim detection\
2.  Claim type classification (explicit vs implicit)\
3.  Claim span extraction (BIO tagging)

The system is trained on Reddit health data and evaluated zero-shot on
the IBM Debater claim benchmark.

Pretrained model weights are included. Retraining is **not required** to
reproduce results.

------------------------------------------------------------------------

## Setup

Tested with Python 3.10.

Create and activate a virtual environment:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

If installation fails, ensure Python 3.10 is used:

``` bash
/opt/homebrew/bin/python3.10 -m venv venv
```

------------------------------------------------------------------------

## Project Structure

- `src/` - core models and pipeline implementation  
- `analysis/` - linguistic analysis and error inspection  
- `ablation_analysis/` - input granularity experiments  
- `benchmark_analysis/` - IBM zero-shot evaluation  
- `experiments/` - pretrained model weights  

------------------------------------------------------------------------

## Quick Test (Entry Point)

Run:

``` bash
python -m src.run_sample
```

Expected output (example):

    Text:
    "I think this medication causes headaches in some people."

    Prediction:
    {
      'claim': True,
      'claim_confidence': 0.87,
      'claim_type': 'explicit',
      'claim_type_confidence': 0.79,
      'span': 'this medication causes headaches'
    }

This confirms that:
- Models load correctly
- Pipeline runs end-to-end
- Span extraction works

------------------------------------------------------------------------

## Zero-Shot Evaluation (IBM Benchmark)

Run:

``` bash
python -m benchmark_analysis.eval_ibm_claim_detection
```

Expected output:

    Running zero-shot evaluation on 2500 sentences...

    IBM Claim Detection (Zero-shot)
    Precision: 0.379
    Recall:    0.285
    F1:        0.326
    Accuracy:  0.654

    Confidence
    Avg (correct): 0.417
    Avg (wrong):   0.469

This evaluates cross-domain generalization (no fine-tuning).

------------------------------------------------------------------------

## Input Granularity Ablation

``` bash
python -m ablation_analysis.run_input_granularity_ablation
```

Expected output:

    Input Granularity Ablation

    Variant              P        R        F1
    ---------------------------------------------
    Full Post            0.801    0.928    0.860
    Any Sentence         0.831    0.740    0.783
    Best Sentence        0.868    0.263    0.404

This compares inference strategies while keeping the model fixed.

------------------------------------------------------------------------

## Claim Typology Analysis

``` bash
python -m analysis.run_claim_typology_corpus
```

Expected output:

    Claim type distribution:
    contrastive: 204
    causal: 195
    epistemic: 110
    normative: 15

    Multi-label stats:
    none: 274
    multi_label: 126

------------------------------------------------------------------------

## Hedging Analysis

``` bash
python -m analysis.run_hedging_corpus
```

Expected output:

    EXPLICIT CLAIMS
    Total spans: 597
    With hedging: 114 (19.10%)
    Avg hedges per span: 0.23

    IMPLICIT CLAIMS
    Total spans: 53
    With hedging: 10 (18.87%)
    Avg hedges per span: 0.25

------------------------------------------------------------------------

## Error Collection

``` bash
python -m analysis.run_error_collection
```

Expected output:

    Saved 266 error cases to analysis/error_samples.csv

All gold disagreements are exported for qualitative analysis.

------------------------------------------------------------------------

## Training (Optional)

Training scripts are included for completeness but are not required to reproduce the reported results, as pretrained weights are already provided.

Pretrained model weights are included in the repository under `experiments/`.

If desired, models can be retrained with:

``` bash
python -m src.train_claim_detection
python -m src.train_claim_type
python -m src.train_claim_span
```
