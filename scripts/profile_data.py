import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    HPO_DISEASE_PHENOTYPE_FILE, HPO_GENE_PHENOTYPE_FILE, 
    CLINVAR_CLEAN_FILE, GWAS_CLEAN_FILE, RELATIONSHIPS_FILE,
    DATASET_PROFILE_FILE, ENTITY_SUMMARY_CSV
)

def profile_dataset(name, df):
    if df.empty:
        return {"Dataset": name, "Rows": 0, "Columns": 0}
        
    return {
        "Dataset": name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing %": round(df.isna().mean().mean() * 100, 2),
        "Duplicates": df.duplicated().sum()
    }

def main():
    print("Profiling datasets...")
    
    profiles = []
    entity_counts = {
        "Variants": set(),
        "Genes": set(),
        "Diseases": set(),
        "Phenotypes": set(),
        "Traits": set(),
        "Chromosomes": set()
    }
    
    if HPO_DISEASE_PHENOTYPE_FILE.exists():
        df = pd.read_parquet(HPO_DISEASE_PHENOTYPE_FILE)
        profiles.append(profile_dataset("HPO Disease-Phenotype", df))
        if 'database_id' in df: entity_counts["Diseases"].update(df['database_id'].dropna())
        if 'hpo_id' in df: entity_counts["Phenotypes"].update(df['hpo_id'].dropna())
        
    if HPO_GENE_PHENOTYPE_FILE.exists():
        df = pd.read_parquet(HPO_GENE_PHENOTYPE_FILE)
        profiles.append(profile_dataset("HPO Gene-Phenotype", df))
        if 'gene_symbol' in df: entity_counts["Genes"].update(df['gene_symbol'].dropna())
        if 'hpo_id' in df: entity_counts["Phenotypes"].update(df['hpo_id'].dropna())
        
    if CLINVAR_CLEAN_FILE.exists():
        df = pd.read_parquet(CLINVAR_CLEAN_FILE)
        profiles.append(profile_dataset("ClinVar", df))
        if 'variation_id' in df: entity_counts["Variants"].update(df['variation_id'].dropna())
        if 'gene' in df: entity_counts["Genes"].update(df['gene'].dropna())
        if 'condition' in df: entity_counts["Diseases"].update(df['condition'].dropna())
        if 'chromosome' in df: entity_counts["Chromosomes"].update(df['chromosome'].dropna())
        
    if GWAS_CLEAN_FILE.exists():
        df = pd.read_parquet(GWAS_CLEAN_FILE)
        profiles.append(profile_dataset("GWAS Catalog", df))
        if 'rs_id' in df: entity_counts["Variants"].update(df['rs_id'].dropna())
        if 'mapped_gene' in df: entity_counts["Genes"].update(df['mapped_gene'].dropna())
        if 'trait' in df: entity_counts["Traits"].update(df['trait'].dropna())
        if 'chromosome' in df: entity_counts["Chromosomes"].update(df['chromosome'].dropna())
        
    if RELATIONSHIPS_FILE.exists():
        df = pd.read_parquet(RELATIONSHIPS_FILE)
        profiles.append(profile_dataset("Relationships", df))
        
    profile_df = pd.DataFrame(profiles)
    profile_df.to_parquet(DATASET_PROFILE_FILE, index=False)
    profile_df.to_csv(DATASET_PROFILE_FILE.with_suffix('.csv'), index=False)
    
    print("\nDataset Profiles:")
    print(profile_df.to_string(index=False))
    
    summary = [{"entity_type": k, "count": len(v)} for k, v in entity_counts.items()]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(ENTITY_SUMMARY_CSV, index=False)
    
    print("\nEntity Counts:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
