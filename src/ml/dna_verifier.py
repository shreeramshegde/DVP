import re
from collections import Counter

VALID_BASES = set("ACGTN")
STRICT_BASES = set("ACGT")

def clean_dna_sequence(raw_text: str) -> str:
    """
    Cleans raw sequence input by stripping FASTA headers,
    whitespace, line breaks, and normalizing to uppercase.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
        
    lines = raw_text.strip().splitlines()
    # Strip FASTA header lines starting with '>'
    seq_lines = [line.strip() for line in lines if not line.strip().startswith(">")]
    cleaned = "".join(seq_lines).upper()
    # Remove internal spaces or formatting tabs
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned

def verify_dna_sequence(raw_text: str, min_length: int = 20) -> dict:
    """
    Validates a candidate DNA sequence, verifies biological nucleotide characters,
    and calculates sequence quality and composition metrics.
    """
    cleaned = clean_dna_sequence(raw_text)
    
    if not cleaned:
        return {
            "is_valid": False,
            "error_message": "Sequence is empty. Please enter or paste a valid DNA nucleotide sequence.",
            "cleaned_sequence": "",
            "length": 0,
            "invalid_characters": []
        }
        
    # Check invalid characters
    invalid_chars = []
    for idx, char in enumerate(cleaned):
        if char not in VALID_BASES:
            invalid_chars.append({"char": char, "index": idx + 1})
            
    if invalid_chars:
        unique_invalid = sorted(list(set(item["char"] for item in invalid_chars)))
        first_few = [f"'{item['char']}' at position {item['index']}" for item in invalid_chars[:5]]
        example_str = ", ".join(first_few)
        if len(invalid_chars) > 5:
            example_str += f", and {len(invalid_chars) - 5} more"
            
        return {
            "is_valid": False,
            "error_message": (
                f"Invalid DNA sequence! Found illegal non-nucleotide character(s): {', '.join(unique_invalid)}. "
                f"Examples: {example_str}. Valid DNA sequences must contain only nucleotides (A, C, G, T)."
            ),
            "cleaned_sequence": cleaned,
            "length": len(cleaned),
            "invalid_characters": unique_invalid
        }
        
    if len(cleaned) < min_length:
        return {
            "is_valid": False,
            "error_message": (
                f"Sequence is too short ({len(cleaned)} bp). Minimum required length is {min_length} bp "
                "to perform reliable k-mer feature extraction and disease classification."
            ),
            "cleaned_sequence": cleaned,
            "length": len(cleaned),
            "invalid_characters": []
        }
        
    # Compute sequence properties
    counts = Counter(cleaned)
    length = len(cleaned)
    a_pct = round((counts["A"] / length) * 100, 2)
    c_pct = round((counts["C"] / length) * 100, 2)
    g_pct = round((counts["G"] / length) * 100, 2)
    t_pct = round((counts["T"] / length) * 100, 2)
    n_pct = round((counts["N"] / length) * 100, 2)
    
    gc_pct = round(((counts["G"] + counts["C"]) / length) * 100, 2)
    at_pct = round(((counts["A"] + counts["T"]) / length) * 100, 2)
    
    purines = counts["A"] + counts["G"]
    pyrimidines = counts["C"] + counts["T"]
    ratio = round(purines / pyrimidines, 2) if pyrimidines > 0 else float("inf")
    
    return {
        "is_valid": True,
        "error_message": None,
        "cleaned_sequence": cleaned,
        "length": length,
        "gc_content": gc_pct,
        "at_content": at_pct,
        "nucleotide_counts": dict(counts),
        "nucleotide_percentages": {
            "A": a_pct,
            "C": c_pct,
            "G": g_pct,
            "T": t_pct,
            "N": n_pct
        },
        "purine_pyrimidine_ratio": ratio,
        "has_ambiguous_bases": counts["N"] > 0
    }
