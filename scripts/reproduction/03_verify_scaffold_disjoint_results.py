#!/usr/bin/env python3
"""
Verify Scaffold-Disjoint Results

Type: RECONSTRUCTED_REPRODUCTION_SCRIPT
Purpose: Verify scaffold-disjoint generalization evaluation
Input: results/scaffold_disjoint/
Output: Summary statistics and leakage verification
"""

import pandas as pd
import numpy as np

EXPECTED_MEAN = 0.7850
EXPECTED_SD = 0.0508

def main():
    # Load results
    results = pd.read_csv('results/scaffold_disjoint/SCAFFOLD_SPLIT_RESULTS_FINAL.csv')
    leakage = pd.read_csv('results/scaffold_disjoint/scaffold_public_group_leakage_audit.csv')

    print("SCAFFOLD-DISJOINT EVALUATION VERIFICATION")
    print("=" * 60)
    print(f"Total splits: {len(results)}")
    print(f"Seeds: {sorted(results['seed'].unique())}")
    print()

    # Verify summary statistics
    mean_auc = results['auc'].mean()
    sd_auc = results['auc'].std(ddof=1)
    median_auc = results['auc'].median()
    min_auc = results['auc'].min()
    max_auc = results['auc'].max()

    print(f"{'Statistic':<20} {'Paper':<12} {'Reproduced':<15} {'Status'}")
    print("-" * 60)
    print(f"{'Mean AUC':<20} {EXPECTED_MEAN:<12.4f} {mean_auc:<15.10f} {'MATCH' if abs(mean_auc-EXPECTED_MEAN)<0.0001 else 'MISMATCH'}")
    print(f"{'SD AUC':<20} {EXPECTED_SD:<12.4f} {sd_auc:<15.10f} {'MATCH' if abs(sd_auc-EXPECTED_SD)<0.0001 else 'MISMATCH'}")
    print(f"{'Median AUC':<20} {'':<12} {median_auc:<15.10f} {''}")
    print(f"{'Range':<20} {'':<12} {min_auc:.4f}-{max_auc:.4f} {''}")
    print()

    # Verify leakage
    print("Leakage Audit (Anonymous Group IDs):")
    print("-" * 60)
    all_pass = (leakage['status'] == 'PASS').all()
    max_scf = leakage['scaffold_group_overlap'].max()
    max_mol = leakage['molecule_group_overlap'].max()
    max_ssg = leakage['structure_solvent_group_overlap'].max()

    print(f"All splits pass: {all_pass}")
    print(f"Max scaffold group overlap: {max_scf}")
    print(f"Max molecule group overlap: {max_mol}")
    print(f"Max structure-solvent group overlap: {max_ssg}")
    print()

    mean_match = abs(mean_auc - EXPECTED_MEAN) < 0.0001
    sd_match = abs(sd_auc - EXPECTED_SD) < 0.0001

    if mean_match and sd_match and all_pass:
        print("STATUS: ALL RESULTS AND CONSTRAINTS VERIFIED")
        return 0
    else:
        print("STATUS: ISSUE DETECTED")
        return 1

if __name__ == '__main__':
    exit(main())
