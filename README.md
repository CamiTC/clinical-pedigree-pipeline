# Automated Family History Extraction and Pedigree Chart Generation

An end-to-end pipeline that extracts structured family history from clinical genetics transcripts using a hybrid NLP-LLM model and generates pedigree charts from the extracted output.

## Pipeline

1. **Preprocessing**: Transcripts are cleaned, segmented by speaker and split into sentences.
2. **NLP Extraction**: Family members, conditions and attributes are extracted using rule-based patterns, BioBERT and medspaCy.
3. **LLM Refinement**: Low-confidence records are refined using GPT-4o.
4. **Pedigree Generation**: Pedigree charts are generated using Pedixplorer (R).
5. **Evaluation**: Extraction output is evaluated against the ground truth using precision, recall and F1.
6. **Graph Analysis**: Graphs are created from the evaluation results for analysis.

## Requirements
Python 3.9–3.12

### Python
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### R
Install R (https://www.r-project.org/), then install the Pedixplorer package:
```r
install.packages("BiocManager")
BiocManager::install("Pedixplorer")
```

### Environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the pipeline

Run the full pipeline:
```bash
python main.py
```

Run individual steps:
```python
from main import run_preprocessing, run_nlp, run_llm, run_pedigrees
```

## Project structure

```
clinical-pedigree-pipeline/
├── data/
│   ├── transcripts/        # Raw transcripts (.txt)
│   └── ground_truth/       # Ground truth file (.csv)
├── outputs/
│   ├── preprocessed/       # Preprocessed transcripts (.csv)
│   ├── nlp/                # NLP extraction output (.csv)
│   ├── llm/                # LLM refinement output (.csv)
│   ├── pedigrees/          # Generated pedigree charts (.png)
│   └── evaluation/         # Evaluation results (.csv)
├── main.py                 # Pipeline entry point
├── preprocessing.py
├── nlp_extraction.py
├── llm_refinement.py
├── pedigree_generation.py
├── draw_pedigree.R
├── evaluation.py
├── graph_analysis.py
└── requirements.txt
```
