import json
import time
import requests
import pandas as pd
from src.config import TARGET_GENES, CLINVAR_RAW_FILE, CLINVAR_CLEAN_FILE

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def fetch_clinvar_for_gene(gene):
    # Search for variants in the gene
    params = {
        "db": "clinvar",
        "term": f"{gene}[gene]",
        "retmode": "json",
        "retmax": 500  # limit to 500 per gene for the dashboard to keep it manageable
    }
    response = requests.get(ESEARCH_URL, params=params)
    response.raise_for_status()
    data = response.json()
    
    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []
        
    # Get summaries
    summary_params = {
        "db": "clinvar",
        "id": ",".join(id_list),
        "retmode": "json"
    }
    summary_response = requests.post(ESUMMARY_URL, data=summary_params)
    summary_response.raise_for_status()
    summary_data = summary_response.json()
    
    records = []
    result = summary_data.get("result", {})
    for uid in id_list:
        record = result.get(uid, {})
        if record:
            records.append(record)
    
    return records

def process_clinvar(force_refresh=False):
    if not force_refresh and CLINVAR_CLEAN_FILE.exists():
        print("ClinVar clean parquet already exists, skipping re-download.")
        return pd.read_parquet(CLINVAR_CLEAN_FILE)
        
    if not force_refresh and CLINVAR_RAW_FILE.exists():
        print("ClinVar raw JSON exists, cleaning from cached raw data...")
        with open(CLINVAR_RAW_FILE, "r") as f:
            all_records = json.load(f)
        return clean_clinvar(all_records)

    all_records = []
    for gene in TARGET_GENES:
        print(f"Fetching ClinVar records for {gene}...")
        records = fetch_clinvar_for_gene(gene)
        for r in records:
            r['_query_gene'] = gene
        all_records.extend(records)
        time.sleep(1) # respect rate limit
        
    with open(CLINVAR_RAW_FILE, "w") as f:
        json.dump(all_records, f)
        
    return clean_clinvar(all_records)

def clean_clinvar(records):
    cleaned = []
    for r in records:
        if 'error' in r:
            continue
            
        genes_list = r.get("genes", [])
        gene_symbol = genes_list[0].get("symbol") if genes_list else r.get("_query_gene")
        
        # Extract conditions from trait_set or germline_classification
        conditions = []
        if r.get("trait_set"):
            conditions.extend([t.get("trait_name") for t in r.get("trait_set", []) if t.get("trait_name")])
        gc = r.get("germline_classification") or {}
        if isinstance(gc, dict) and gc.get("trait_set"):
            conditions.extend([t.get("trait_name") for t in gc.get("trait_set", []) if t.get("trait_name")])
            
        valid_conditions = [c.strip() for c in conditions if c and c.strip().lower() not in ["not provided", "not specified"]]
        condition = valid_conditions[0] if valid_conditions else "Not provided"
        
        # Extract clinical significance and review status
        sig = gc.get("description") or (r.get("clinical_significance", {}) or {}).get("description", "Not provided")
        review_status = gc.get("review_status") or (r.get("clinical_significance", {}) or {}).get("review_status", "Not provided")
        
        variation_set = r.get("variation_set", [])
        var_type = variation_set[0].get("variation_name", "") if variation_set else ""
        
        cleaned.append({
            "variation_id": str(r.get("uid", "")),
            "accession": r.get("accession", ""),
            "gene": gene_symbol,
            "condition": condition,
            "clinical_significance": sig,
            "review_status": review_status,
            "chromosome": r.get("chr", ""),
            "position": r.get("start", ""),
            "variant_type": var_type
        })
        
    df = pd.DataFrame(cleaned)
    df = df.drop_duplicates(subset=["variation_id"])
    df.to_parquet(CLINVAR_CLEAN_FILE, index=False)
    return df
