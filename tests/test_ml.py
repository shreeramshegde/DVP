import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.ml.dna_verifier import verify_dna_sequence, clean_dna_sequence
from src.ml.classifier import train_and_evaluate_model, predict_and_justify, MODEL_FILE, METRICS_FILE

def test_dna_verifier_valid():
    # Canonical valid DNA sequence
    seq = "ATGCGATCGATCGATCGATCGAATCGATCGATC"
    res = verify_dna_sequence(seq)
    assert res["is_valid"] is True
    assert res["error_message"] is None
    assert res["length"] == len(seq)
    assert 0 <= res["gc_content"] <= 100
    assert 0 <= res["at_content"] <= 100
    assert set(res["nucleotide_counts"].keys()).issubset({"A", "C", "G", "T", "N"})

def test_dna_verifier_fasta_and_spaces():
    raw_fasta = ">sample_fasta_header\nATG C GA T C G ATC GA\nTC GATC GA ATC GAT CG ATC\n"
    res = verify_dna_sequence(raw_fasta)
    assert res["is_valid"] is True
    assert ">" not in res["cleaned_sequence"]
    assert " " not in res["cleaned_sequence"]

def test_dna_verifier_invalid_characters():
    # Sequence with illegal chars (1, 9, X, Z)
    invalid_seq = "ATGCXGAT19ZATCGAATCGATC"
    res = verify_dna_sequence(invalid_seq)
    assert res["is_valid"] is False
    assert res["error_message"] is not None
    assert any(c in res["invalid_characters"] for c in ["X", "1", "9", "Z"])

def test_dna_verifier_too_short():
    short_seq = "ATGCGATC"
    res = verify_dna_sequence(short_seq, min_length=20)
    assert res["is_valid"] is False
    assert "too short" in res["error_message"].lower()

def test_ml_model_and_prediction():
    assert MODEL_FILE.exists()
    assert METRICS_FILE.exists()
    
    # Test sample sequence
    test_seq = "ATGGAGAGCGACGAGAGCGGCCTGCCCGCCATGGAGATCGAGTGCCGCATCACCGGCACCCTGAACGGCGTGGAGTTCGAGCTGGTGGGCGGCGGAGAGGGCACCCCCGAGCAGGGCCGCATGACCAACAAGATGAAGAGCACCAAAGGCGCCCTGACCTTCAGCCCCTA"
    res = predict_and_justify(test_seq)
    
    assert "predicted_disease" in res
    assert "confidence_score" in res
    assert 0.0 <= res["confidence_score"] <= 1.0
    assert "candidate_diseases" in res
    assert len(res["candidate_diseases"]) > 0
    assert "driving_motifs" in res
    assert len(res["driving_motifs"]) > 0
    assert "biological_context" in res
    assert "justification_summary" in res
    assert len(res["justification_summary"]) > 20

def test_data_leakage_guarantee():
    model, metrics = train_and_evaluate_model()
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0
    assert metrics["accuracy"] >= 0.85
    assert metrics["f1_weighted"] >= 0.85
    # Ensure vectorizer vocabulary is encapsulated in pipeline
    assert "tfidf" in model.named_steps
    assert "clf" in model.named_steps
