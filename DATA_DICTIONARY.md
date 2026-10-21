# Data Dictionary

## `clinvar_clean.parquet`
| column | type | description | source | transformation |
|--------|------|-------------|--------|----------------|
| variation_id | string | ClinVar Variation ID | ClinVar | Extracted from uid |
| accession | string | ClinVar Accession | ClinVar | Direct mapping |
| gene | string | Gene symbol | ClinVar | Extracted from genes list or query |
| condition | string | Associated condition | ClinVar | Extracted from trait_set |
| clinical_significance | string | Classification | ClinVar | Extracted from description |
| review_status | string | Review status | ClinVar | Extracted from significance info |
| chromosome | string | Chromosome | ClinVar | Direct mapping |
| position | string | Genomic position | ClinVar | Direct mapping |
| variant_type | string | Type of variant | ClinVar | Extracted from variation_set |

## `gwas_associations.parquet`
| column | type | description | source | transformation |
|--------|------|-------------|--------|----------------|
| rs_id | string | dbSNP identifier | GWAS Catalog | Extracted from SNP search |
| variant_identifier | string | Variant identifier | GWAS Catalog | Duplicated from rs_id |
| trait | string | EFO trait name | GWAS Catalog | Linked from association EFO traits |
| mapped_gene | string | Mapped Gene | GWAS Catalog | Query parameter |
| chromosome | string | Chromosome | GWAS Catalog | Extracted from SNP locations |
| position | string | Position | GWAS Catalog | Extracted from SNP locations |
| p_value | float | P-value | GWAS Catalog | Direct mapping |
| study_accession | string | Study Accession | GWAS Catalog | Linked from association study |

## `human_os_relationships.parquet`
| column | type | description | source | transformation |
|--------|------|-------------|--------|----------------|
| source_id | string | Source node ID | Multiple | Identifier |
| source_name | string | Source node display name | Multiple | Name or identifier |
| source_type | string | Type of source entity | Multiple | e.g. Gene, Variant |
| relationship | string | Directed edge label | Multiple | Hardcoded based on edge source |
| target_id | string | Target node ID | Multiple | Identifier |
| target_name | string | Target node display name | Multiple | Name or identifier |
| target_type | string | Type of target entity | Multiple | e.g. Phenotype, Condition |
| dataset_source | string | Originating database | Multiple | HPO, ClinVar, or GWAS |
