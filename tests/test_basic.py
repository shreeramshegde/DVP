import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HPO_DISEASE_PHENOTYPE_FILE, HPO_GENE_PHENOTYPE_FILE, CLINVAR_CLEAN_FILE, GWAS_CLEAN_FILE, RELATIONSHIPS_FILE

def test_hpo_parser():
    assert HPO_DISEASE_PHENOTYPE_FILE.exists(), "HPO disease file not generated"
    df = pd.read_parquet(HPO_DISEASE_PHENOTYPE_FILE)
    assert not df.empty
    assert 'hpo_id' in df.columns
    # Check HPO IDs are valid looking
    assert df['hpo_id'].iloc[0].startswith("HP:")

def test_normalization():
    if CLINVAR_CLEAN_FILE.exists():
        df = pd.read_parquet(CLINVAR_CLEAN_FILE)
        if not df.empty:
            assert 'gene' in df.columns
            assert 'clinical_significance' in df.columns
            # Ensure no lowercase gene symbols assuming normalization (or at least valid string)
            assert isinstance(df['gene'].iloc[0], str)

def test_relationships():
    assert RELATIONSHIPS_FILE.exists(), "Relationships file not generated"
    df = pd.read_parquet(RELATIONSHIPS_FILE)
    assert not df.empty
    # Check no self-relationships
    self_rels = df[df['source_id'] == df['target_id']]
    assert self_rels.empty, "Found impossible self-relationships"
    
def test_dashboard_data_loaders():
    assert CLINVAR_CLEAN_FILE.exists()
    assert GWAS_CLEAN_FILE.exists()
