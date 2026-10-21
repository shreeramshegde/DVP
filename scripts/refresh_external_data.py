import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_data import main as prep_main
from scripts.profile_data import main as profile_main

def main():
    print("--- HUMAN OS Data Refresh ---")
    print("Refreshing from ClinVar and GWAS Catalog...")
    # Just run the prep script since it handles the downloading/caching logic natively
    prep_main()
    print("\nRe-generating Data Profiles...")
    profile_main()
    print("--- Refresh Complete ---")

if __name__ == "__main__":
    main()
