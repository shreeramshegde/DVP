import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hpo_loader import process_hpo_phenotype, process_hpo_genes
from src.data.clinvar_loader import process_clinvar
from src.data.gwas_loader import process_gwas
from src.relationships.relationship_builder import build_relationships

def main():
    print("Starting data preparation pipeline...")
    
    print("Processing HPO Disease-Phenotype annotations...")
    process_hpo_phenotype()
    
    print("Processing HPO Gene-Phenotype annotations...")
    process_hpo_genes()
    
    print("Processing ClinVar targeted queries...")
    process_clinvar()
    
    print("Processing GWAS Catalog targeted queries...")
    process_gwas()
    
    print("Building knowledge relationships...")
    build_relationships()
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
