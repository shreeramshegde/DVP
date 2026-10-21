import pandas as pd
from src.config import (
    HPO_DISEASE_PHENOTYPE_FILE, HPO_GENE_PHENOTYPE_FILE, 
    CLINVAR_CLEAN_FILE, GWAS_CLEAN_FILE, RELATIONSHIPS_FILE
)

def build_relationships():
    relationships = []
    
    # 1. HPO Disease to Phenotype
    if HPO_DISEASE_PHENOTYPE_FILE.exists():
        df_dis_pheno = pd.read_parquet(HPO_DISEASE_PHENOTYPE_FILE)
        for _, row in df_dis_pheno.iterrows():
            # some basic validation
            if pd.isna(row.get('database_id')) or pd.isna(row.get('hpo_id')):
                continue
            relationships.append({
                "source_id": row['database_id'],
                "source_name": row.get('disease_name', row['database_id']),
                "source_type": "Disease/Condition",
                "relationship": "has_phenotype",
                "target_id": row['hpo_id'],
                "target_name": row['hpo_id'], # ideally we'd look up the name if available
                "target_type": "Phenotype",
                "dataset_source": "HPO"
            })
            
    # 2. HPO Gene to Phenotype
    if HPO_GENE_PHENOTYPE_FILE.exists():
        df_gene_pheno = pd.read_parquet(HPO_GENE_PHENOTYPE_FILE)
        for _, row in df_gene_pheno.iterrows():
            if pd.isna(row.get('gene_symbol')) or pd.isna(row.get('hpo_id')):
                continue
            relationships.append({
                "source_id": row['gene_symbol'],
                "source_name": row['gene_symbol'],
                "source_type": "Gene",
                "relationship": "associated_with_phenotype",
                "target_id": row['hpo_id'],
                "target_name": row.get('hpo_name', row['hpo_id']),
                "target_type": "Phenotype",
                "dataset_source": "HPO"
            })
            
    # 3. ClinVar Variant to Gene & Condition
    if CLINVAR_CLEAN_FILE.exists():
        df_clinvar = pd.read_parquet(CLINVAR_CLEAN_FILE)
        for _, row in df_clinvar.iterrows():
            var_id = str(row['variation_id'])
            gene = row['gene']
            cond = row['condition']
            
            if pd.notna(gene) and gene != "":
                relationships.append({
                    "source_id": var_id,
                    "source_name": var_id,
                    "source_type": "Variant",
                    "relationship": "located_in",
                    "target_id": gene,
                    "target_name": gene,
                    "target_type": "Gene",
                    "dataset_source": "ClinVar"
                })
                
            if pd.notna(cond) and cond != "Not provided":
                relationships.append({
                    "source_id": var_id,
                    "source_name": var_id,
                    "source_type": "Variant",
                    "relationship": "associated_with_condition",
                    "target_id": cond,
                    "target_name": cond,
                    "target_type": "Condition",
                    "dataset_source": "ClinVar"
                })
                
    # 4. GWAS Variant to Gene & Trait
    if GWAS_CLEAN_FILE.exists():
        df_gwas = pd.read_parquet(GWAS_CLEAN_FILE)
        for _, row in df_gwas.iterrows():
            rs_id = row['rs_id']
            gene = row['mapped_gene']
            trait = row['trait']
            
            if pd.notna(gene) and gene != "":
                relationships.append({
                    "source_id": rs_id,
                    "source_name": rs_id,
                    "source_type": "Variant",
                    "relationship": "mapped_to",
                    "target_id": gene,
                    "target_name": gene,
                    "target_type": "Gene",
                    "dataset_source": "GWAS Catalog"
                })
                
            if pd.notna(trait) and trait != "Unknown":
                relationships.append({
                    "source_id": rs_id,
                    "source_name": rs_id,
                    "source_type": "Variant",
                    "relationship": "associated_with_trait",
                    "target_id": trait,
                    "target_name": trait,
                    "target_type": "Trait",
                    "dataset_source": "GWAS Catalog"
                })

    df_rels = pd.DataFrame(relationships)
    df_rels = df_rels.drop_duplicates()
    df_rels.to_parquet(RELATIONSHIPS_FILE, index=False)
    return df_rels
