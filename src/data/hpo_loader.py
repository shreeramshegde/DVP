import pandas as pd
from src.config import HPO_PHENOTYPE_FILE, HPO_GENE_FILE, HPO_DISEASE_PHENOTYPE_FILE, HPO_GENE_PHENOTYPE_FILE

def process_hpo_phenotype():
    if not HPO_PHENOTYPE_FILE.exists():
        raise FileNotFoundError(f"{HPO_PHENOTYPE_FILE} not found.")
    
    # Read phenotype.hpoa, skipping metadata comments but finding the header
    with open(HPO_PHENOTYPE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("database_id"):
            start_idx = i
            break
            
    df = pd.read_csv(
        HPO_PHENOTYPE_FILE, 
        sep='\t', 
        skiprows=start_idx, 
        dtype=str
    )
    
    df.to_parquet(HPO_DISEASE_PHENOTYPE_FILE, index=False)
    return df

def process_hpo_genes():
    if not HPO_GENE_FILE.exists():
        raise FileNotFoundError(f"{HPO_GENE_FILE} not found.")
        
    df = pd.read_csv(
        HPO_GENE_FILE, 
        sep='\t', 
        comment='#',
        dtype=str
    )
    
    df.to_parquet(HPO_GENE_PHENOTYPE_FILE, index=False)
    return df
