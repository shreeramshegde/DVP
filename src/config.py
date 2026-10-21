import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GWAS_CACHE_DIR = RAW_DIR / "gwas_cache"

HPO_PHENOTYPE_FILE = RAW_DIR / "phenotype.hpoa"
HPO_GENE_FILE = RAW_DIR / "genes_to_phenotype.txt"
CLINVAR_RAW_FILE = RAW_DIR / "clinvar_targeted.json"

CLINVAR_CLEAN_FILE = PROCESSED_DIR / "clinvar_clean.parquet"
GWAS_CLEAN_FILE = PROCESSED_DIR / "gwas_associations.parquet"
HPO_DISEASE_PHENOTYPE_FILE = PROCESSED_DIR / "hpo_disease_phenotype.parquet"
HPO_GENE_PHENOTYPE_FILE = PROCESSED_DIR / "hpo_gene_phenotype.parquet"
RELATIONSHIPS_FILE = PROCESSED_DIR / "human_os_relationships.parquet"
DATASET_PROFILE_FILE = PROCESSED_DIR / "dataset_profile.parquet"
ENTITY_SUMMARY_CSV = PROCESSED_DIR / "entity_summary.csv"

TARGET_GENES = [
    "BRCA1",
    "BRCA2",
    "TP53",
    "CFTR",
    "LDLR",
    "HBB",
    "APOB",
    "PCSK9"
]
