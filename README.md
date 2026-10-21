# HUMAN OS — Interactive Genomic Variant and Disease Visualization

## 1. Project Description
HUMAN OS is an interactive genomic data visualization system. It allows users to explore genetic variants, genes, phenotypes, diseases, conditions, and GWAS traits using multiple real biomedical data sources.

## 2. Problem Statement
Biomedical data is often fragmented across multiple databases (HPO, ClinVar, GWAS Catalog) with different structures, identifiers, and biological meanings. This project demonstrates how heterogeneous biomedical datasets can be cleaned, normalized, connected where justified, and transformed into interactive visualizations without falsely inflating causal relationships.

## 3. Objectives
- Profile and integrate targeted biomedical datasets.
- Provide a responsive offline-first dashboard.
- Display relationships between genes, variants, and phenotypes.
- Preserve source provenance and distinct ontology semantics.

## 4. Data Sources
- **Human Phenotype Ontology (HPO):** `phenotype.hpoa` and `genes_to_phenotype.txt`
- **NCBI ClinVar:** Targeted API queries via E-utilities.
- **NHGRI-EBI GWAS Catalog:** Targeted API queries via REST API.

## 5. Architecture
1. **Raw Data / API**: Data is fetched and cached locally.
2. **Data Pipeline**: Python scripts parse, clean, normalize, and deduplicate data.
3. **Relationships**: An explicit relationship table connects exact identifiers.
4. **Dashboard**: A Streamlit application visualizes the `processed/*.parquet` files.

## 6. Folder Structure
- `data/raw/`: Original downloaded files and API caches.
- `data/processed/`: Cleaned Parquet files.
- `src/`: Core Python modules for data loading, normalization, and relationships.
- `scripts/`: Executable scripts for preparation and profiling.
- `tests/`: Basic validation tests.
- `app.py`: Streamlit dashboard entrypoint.

## 7. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 8. Data Preparation
```bash
python scripts/prepare_data.py
python scripts/profile_data.py
```

## 9. Dashboard Launch
```bash
streamlit run app.py
```

## 10. Dashboard Pages
- **Overview**: High-level KPIs and summary charts.
- **Variant Explorer**: Filter and explore ClinVar variants.
- **Gene / Phenotype Explorer**: Cross-dataset view of a specific gene.
- **Disease / Phenotype Explorer**: HPO annotations for diseases.
- **GWAS Explorer**: GWAS traits and associations.
- **Network Explorer**: Interactive neighborhood graph.
- **Data Sources & Methodology**: Data provenance, schemas, and citation details.
- **DNA Sequence Disease Classifier**: Machine Learning sequence verification, multi-class disease prediction, driving k-mer motif attribution, and metrics justification.

## 11. Machine Learning Classification System
HUMAN OS features a complete, interpretable DNA sequence classifier:
- **Biological Sequence Verification:** Validates input DNA sequences against the standard nucleotide alphabet (A, C, G, T), detects illegal non-nucleotide characters, strips FASTA headers/whitespace, and computes GC-content, AT-content, and purine/pyrimidine ratios.
- **Interpretable k-mer TF-IDF Classifier:** Utilizes character n-grams ($k=3$ to $6$) encapsulated within an `sklearn.pipeline.Pipeline` with `LogisticRegression`.
- **Zero Data Leakage Protocol:**
  - Stratified 75/25 split executed strictly *before* any vectorization or feature extraction.
  - Vocabulary and IDF statistics fitted exclusively on the training fold.
  - 5-Fold Stratified Cross-Validation performed on training fold only ($92.5\% \pm 1.6\%$).
  - Evaluated on $N = 317$ held-out test sequences with $95.9\%$ accuracy and $95.9\%$ weighted F1-score.
- **Dual-Layer Decision Justification:**
  - *Mathematical Justification:* Local feature attribution extracting the top contributing k-mer motifs driving the prediction with positive weights.
  - *Biological Grounding:* Chromosomal locus, associated gene(s), inheritance pattern, clinical phenotypes from HPO, and molecular pathogenic mechanism.

## 12. Data Limitations
Data in this dashboard is based on targeted queries for specific genes (e.g., BRCA1, TP53) to remain manageable on a local machine. It does not represent the entirety of human genomic knowledge.

## 13. Scientific Limitations
This project is an educational and exploratory data visualization system. The relationships shown are derived from public biomedical databases and represent reported annotations or associations. They do not by themselves establish causation, individual disease risk, or a medical diagnosis.

## 14. Reproducibility Instructions
To refresh the external API caches and regenerate processed files, run:
```bash
python scripts/refresh_external_data.py
```

## 15. Automated Verification & Testing
To run the automated test suite covering HPO parsing, ClinVar & GWAS normalization, graph relationships, DNA verification, and ML classifier metrics:
```bash
pytest tests/
```
