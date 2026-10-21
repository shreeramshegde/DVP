import os
import time
import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS_CACHE_DIR = BASE_DIR / "data" / "raw" / "transcripts_cache"
ML_DATASET_FILE = BASE_DIR / "data" / "processed" / "ml_sequences.parquet"

# Canonical NCBI RefSeq transcripts and their definitive disease associations
GENE_METADATA = {
    "BRCA1": {
        "accession": "NM_007294.4",
        "disease_class": "Hereditary Breast & Ovarian Cancer",
        "chromosome": "chr17"
    },
    "BRCA2": {
        "accession": "NM_000059.4",
        "disease_class": "Hereditary Breast & Ovarian Cancer",
        "chromosome": "chr13"
    },
    "TP53": {
        "accession": "NM_000546.6",
        "disease_class": "Li-Fraumeni Syndrome",
        "chromosome": "chr17"
    },
    "CFTR": {
        "accession": "NM_000492.4",
        "disease_class": "Cystic Fibrosis",
        "chromosome": "chr7"
    },
    "LDLR": {
        "accession": "NM_000527.5",
        "disease_class": "Familial Hypercholesterolemia",
        "chromosome": "chr19"
    },
    "HBB": {
        "accession": "NM_000518.5",
        "disease_class": "Beta-Thalassemia & Hemoglobinopathy",
        "chromosome": "chr11"
    },
    "APOB": {
        "accession": "NM_000384.3",
        "disease_class": "Familial Hypercholesterolemia",
        "chromosome": "chr2"
    },
    "PCSK9": {
        "accession": "NM_174936.4",
        "disease_class": "Familial Hypercholesterolemia",
        "chromosome": "chr1"
    }
}

def fetch_or_load_transcript(gene: str, accession: str) -> str:
    """
    Retrieves full-length cDNA sequence from local cache or NCBI E-utilities eFetch.
    """
    TRANSCRIPTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = TRANSCRIPTS_CACHE_DIR / f"{gene}_{accession}.fasta"
    
    if cache_file.exists():
        with open(cache_file, "r") as f:
            lines = f.readlines()
        return "".join(l.strip() for l in lines if not l.startswith(">")).upper()
        
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    print(f"Fetching {gene} ({accession}) sequence from NCBI...")
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    
    with open(cache_file, "w") as f:
        f.write(response.text)
        
    lines = response.text.strip().splitlines()
    seq = "".join(l.strip() for l in lines if not l.startswith(">")).upper()
    time.sleep(0.5)
    return seq

def generate_sequence_dataset(window_size: int = 120) -> pd.DataFrame:
    """
    Generates structured labeled sequence fragments from authentic reference cDNA transcripts.
    Uses adaptive step sizing so shorter genes (like HBB, TP53) maintain sufficient sample representation.
    """
    records = []
    for gene, meta in GENE_METADATA.items():
        accession = meta["accession"]
        disease = meta["disease_class"]
        chrom = meta["chromosome"]
        
        full_seq = fetch_or_load_transcript(gene, accession)
        seq_len = len(full_seq)
        
        # Adaptive step size: smaller step for shorter genes to balance class support
        if seq_len < 1000:
            step_size = 8
        elif seq_len < 3000:
            step_size = 20
        elif seq_len < 7000:
            step_size = 35
        else:
            step_size = 50
        
        # Slide windows across transcript
        for start_pos in range(0, seq_len - window_size + 1, step_size):
            frag = full_seq[start_pos : start_pos + window_size]
            records.append({
                "sequence": frag,
                "gene": gene,
                "disease_class": disease,
                "chromosome": chrom,
                "accession": accession,
                "start_pos": start_pos,
                "end_pos": start_pos + window_size
            })
            
        # Ensure tail segment is captured
        if (seq_len - window_size) % step_size != 0:
            frag = full_seq[-window_size:]
            records.append({
                "sequence": frag,
                "gene": gene,
                "disease_class": disease,
                "chromosome": chrom,
                "accession": accession,
                "start_pos": seq_len - window_size,
                "end_pos": seq_len
            })
            
    df = pd.DataFrame(records).drop_duplicates(subset=["sequence"])
    ML_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ML_DATASET_FILE, index=False)
    print(f"Generated {len(df)} sequence samples across {df['disease_class'].nunique()} disease classes.")
    return df

if __name__ == "__main__":
    generate_sequence_dataset()
