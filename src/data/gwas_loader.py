import os
import json
import time
import requests
import pandas as pd
from src.config import TARGET_GENES, GWAS_CACHE_DIR, GWAS_CLEAN_FILE

BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api/v2"

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "HumanOS-Genomic-Dashboard/2.0 (Academic Visualization Project)"
})

def get_cache_path(resource_name):
    # sanitize resource name for filename
    safe_name = resource_name.replace("/", "_").replace("?", "_").replace("=", "_").replace("&", "_")
    return GWAS_CACHE_DIR / f"{safe_name}.json"

def gwas_query(resource, params=None):
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        resource_full = f"{resource}?{query_string}"
    else:
        resource_full = resource
        
    cache_file = get_cache_path(resource_full)
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                if data == "__NOT_FOUND__":
                    return {}
                return data
        except Exception:
            pass
            
    url = f"{BASE_URL}/{resource}"
    print(f"Querying GWAS REST v2 {url} with params {params}")
    for attempt in range(5):
        try:
            response = session.get(url, params=params, timeout=20)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_sec = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
                print(f"Rate limited (429). Waiting for {wait_sec} seconds...")
                time.sleep(wait_sec)
                continue
                
            if response.status_code in (404, 410):
                with open(cache_file, "w") as f:
                    json.dump("__NOT_FOUND__", f)
                return {}
                
            response.raise_for_status()
            data = response.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            time.sleep(0.5) # Polite delay
            return data
        except requests.exceptions.RequestException as e:
            if attempt == 4:
                print(f"Error querying {url}: {e}")
                return None
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected error querying {url}: {e}")
            return None
    return None

def fetch_gwas_for_gene(gene):
    associations = []
    
    # Query GWAS Catalog v2 associations mapped to target gene
    data = gwas_query("associations", {"mappedGene": gene, "size": 50})
    if not data or "_embedded" not in data:
        return associations
        
    assocs = data["_embedded"].get("associations", [])
    for a in assocs:
        rs_id = None
        if a.get("snp_allele"):
            rs_id = a.get("snp_allele")[0].get("rs_id")
        elif a.get("snp_effect_allele"):
            rs_id = a.get("snp_effect_allele")[0].split("-")[0]
            
        if not rs_id:
            continue
            
        traits = [t.get("efo_trait") for t in a.get("efo_traits", []) if t.get("efo_trait")]
        if not traits and a.get("reported_trait"):
            traits = a.get("reported_trait")
        trait_name = traits[0] if traits else "Unknown"
        
        loc = a.get("locations", [""])[0] if a.get("locations") else ""
        chrom = str(loc.split(":")[0]) if ":" in loc else ""
        pos = str(loc.split(":")[1]) if ":" in loc else ""
        
        associations.append({
            "rs_id": rs_id,
            "variant_identifier": rs_id,
            "trait": trait_name,
            "mapped_gene": gene,
            "chromosome": chrom,
            "position": pos,
            "p_value": a.get("p_value"),
            "study_accession": a.get("accession_id", "Unknown")
        })
        
    return associations

def process_gwas():
    if not GWAS_CACHE_DIR.exists():
        os.makedirs(GWAS_CACHE_DIR, exist_ok=True)
        
    all_associations = []
    for gene in TARGET_GENES:
        print(f"Fetching GWAS records for {gene}...")
        assocs = fetch_gwas_for_gene(gene)
        print(f"-> Retrieved {len(assocs)} associations for {gene}")
        all_associations.extend(assocs)
        
    df = pd.DataFrame(all_associations)
    if df.empty:
        df = pd.DataFrame(columns=[
            "rs_id", "variant_identifier", "trait", "mapped_gene", 
            "chromosome", "position", "p_value", "study_accession"
        ])
    else:
        df = df.drop_duplicates()
        
    df.to_parquet(GWAS_CLEAN_FILE, index=False)
    df.to_csv(GWAS_CLEAN_FILE.with_suffix('.csv'), index=False)
    print(f"GWAS processing complete. Total associations saved: {len(df)}")
    return df
