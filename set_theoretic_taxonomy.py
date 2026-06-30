#!/usr/bin/env python3

"""

risk_taxonomy_classifier.py

 

Reference implementation of the Set-Theoretic Risk Taxonomy Classification Framework.

Author: Marc Faria

License: MIT

 

This module provides:

  - Data structures for risks, drivers, and impact matrices

  - Computation of BREADTH, CR (Concentration Ratio), and HHI

  - Attribute meet operation for factorization detection

  - Classification decision rule (majority-of-tests)

 

Usage:

    python risk_taxonomy_classifier.py --input sample_input.json --output results.json

 

Requirements:

    Python 3.7+ (standard library only, no external dependencies)

"""

 

import json

import argparse

from dataclasses import dataclass, field

from typing import Dict, List, Set, Tuple, Optional

from enum import Enum

 

# =============================================================================

# SECTION 1: DATA STRUCTURES

# =============================================================================

 

class Classification(Enum):

    """Classification outcomes for taxonomy candidates."""

    SUB_RISK = "SUB-RISK"

    DRIVER = "DRIVER"

    UNDETERMINED = "UNDETERMINED"

 

@dataclass

class ControlledVocabulary:

    """

    Controlled vocabulary V organized by semantic domain.

   

    Represents the attribute lexicon used for risk and candidate characterization.

    Institutions may extend this vocabulary by adding terms to the domain sets.

    """

    scope: Set[str] = field(default_factory=lambda: {

        "internal", "external", "cross-cutting", "enterprise-wide"

    })

    nature: Set[str] = field(default_factory=lambda: {

        "financial", "operational", "strategic", "behavioral"

    })

    trigger: Set[str] = field(default_factory=lambda: {

        "market-driven", "event-driven", "process-driven", "decision-driven"

    })

   

    @property

    def all_terms(self) -> Set[str]:

        """Return the complete vocabulary as a single set."""

        return self.scope | self.nature | self.trigger

   

    def validate(self, attributes: Set[str]) -> bool:

        """Check that all attributes belong to the vocabulary."""

        return attributes.issubset(self.all_terms)

 

@dataclass

class L1Risk:

    """

    An L1 (top-level) risk category in the enterprise taxonomy.

   

    Attributes:

        name: Identifier for the risk category (e.g., "Credit Risk")

        attributes: Set of vocabulary terms characterizing this risk

        definition: Optional text definition for documentation

    """

    name: str

    attributes: Set[str]

    definition: str = ""

 

@dataclass

class Candidate:

    """

    A candidate concept to be classified as sub-risk or driver.

   

    Attributes:

        name: Identifier for the candidate (e.g., "Geopolitical Risk")

        attributes: Set of vocabulary terms characterizing this candidate

        impact_weights: Dict mapping L1 risk names to impact weights (non-normalized)

        normalized_weights: Computed automatically; weights summing to 1.0

    """

    name: str

    attributes: Set[str]

    impact_weights: Dict[str, float]

   

    def __post_init__(self):

        """Normalize weights to sum to 1.0 after initialization."""

        total = sum(self.impact_weights.values())

        if total > 0:

            self.normalized_weights = {

                k: v / total for k, v in self.impact_weights.items()

            }

        else:

            self.normalized_weights = {}

 

@dataclass

class ClassificationResult:

    """

    Complete result of classifying a candidate.

   

    Contains computed metrics, individual test outcomes, final classification,

    and a human-readable rationale for audit trail purposes.

    """

    candidate_name: str

    breadth: int

    cr: float

    hhi: float

    relevance_test: Classification

    distinction_test: Classification

    materiality_test: Classification

    final_classification: Classification

    parent_risk: Optional[str] = None

    rationale: str = ""

 

# =============================================================================

# SECTION 2: CORE COMPUTATIONS

# =============================================================================

 

def compute_breadth(candidate: Candidate) -> int:

    """

    Compute BREADTH(c) = |{r ∈ L1 : impact(c, r) ≠ ∅}|

   

    Count of L1 risks with non-zero impact weight.

    A value > 1 signals cross-cutting influence.

   

    Args:

        candidate: The candidate to evaluate

       

    Returns:

        Number of L1 categories with positive impact weight

    """

    return sum(1 for w in candidate.normalized_weights.values() if w > 0)

 

def compute_cr(candidate: Candidate) -> float:

    """

    Compute CR(c) = max_i s_i

   

    Concentration Ratio: maximum normalized impact share.

    CR ≥ 0.50 indicates majority of impact on single L1.

   

    Args:

        candidate: The candidate to evaluate

       

    Returns:

        Maximum share value (0.0 to 1.0)

    """

    if not candidate.normalized_weights:

        return 0.0

    return max(candidate.normalized_weights.values())

 

def compute_hhi(candidate: Candidate) -> float:

    """

    Compute HHI(c) = Σ_i s_i^2

   

    Herfindahl-Hirschman Index: sum of squared normalized shares.

    HHI ≥ 0.25 indicates moderately concentrated materiality.

   

    Args:

        candidate: The candidate to evaluate

       

    Returns:

        HHI value (1/n for uniform distribution, 1.0 for single-category concentration)

    """

    return sum(s ** 2 for s in candidate.normalized_weights.values())

 

def attribute_meet(candidates: List[Candidate]) -> Set[str]:

    """

    Compute meet_A({R_i}) = ⋂_i A(R_i)

   

    Returns the intersection of attribute sets across candidates.

    A non-empty meet indicates potential parent-child relationship.

   

    Args:

        candidates: List of candidates to compute meet over

       

    Returns:

        Intersection of all candidate attribute sets

    """

    if not candidates:

        return set()

    result = candidates[0].attributes.copy()

    for c in candidates[1:]:

        result &= c.attributes

    return result

 

def find_factorization_parent(

    candidate: Candidate,

    l1_risks: List[L1Risk]

) -> Optional[L1Risk]:

    """

    Distinction Test: Find R* ∈ R such that A(R*) ⊆ A(c).

   

    Checks for attribute subsumption (parent attributes are subset of candidate)

    AND impact alignment (majority of weight concentrated on the parent).

   

    Args:

        candidate: The candidate to test for factorization

        l1_risks: List of L1 risk categories to check against

       

    Returns:

        The L1 risk through which the candidate factors, or None

    """

    for risk in l1_risks:

        # Check attribute subsumption: parent attributes subset of candidate attributes

        if risk.attributes.issubset(candidate.attributes):

            # Check impact alignment: majority of weight on this risk

            if candidate.normalized_weights.get(risk.name, 0) >= 0.50:

                return risk

    return None

 

# =============================================================================

# SECTION 3: CLASSIFICATION TESTS

# =============================================================================

 

@dataclass

class Thresholds:

    """

    Classification thresholds with documented rationale.

   

    Default values are based on:

    - CR = 0.50: Majority-share principle (>50% concentration indicates dominant parent)

    - HHI = 0.25: Adapted from DOJ/FTC "moderately concentrated" threshold

   

    Institutions should validate these against their own classification benchmarks

    or conduct calibration studies as described in the manuscript.

    """

    cr_threshold: float = 0.50

    hhi_threshold: float = 0.25

   

    # Metadata for audit trail

    cr_rationale: str = "Majority-share principle: >50% concentration indicates dominant parent"

    hhi_rationale: str = "Adapted from DOJ/FTC 'moderately concentrated' threshold (0.25)"

    calibration_source: str = "DOJ/FTC Horizontal Merger Guidelines (2010); validated via sensitivity analysis"

 

def run_relevance_test(candidate: Candidate, thresholds: Thresholds) -> Classification:

    """

    Relevance Test: CR(c) ≥ threshold → SUB-RISK

   

    Tests whether the candidate concentrates majority of impact on single L1.

   

    Args:

        candidate: The candidate to test

        thresholds: Threshold configuration

       

    Returns:

        SUB-RISK if CR meets threshold, DRIVER otherwise

    """

    cr = compute_cr(candidate)

    return Classification.SUB_RISK if cr >= thresholds.cr_threshold else Classification.DRIVER

 

def run_distinction_test(

    candidate: Candidate,

    l1_risks: List[L1Risk]

) -> Tuple[Classification, Optional[str]]:

    """

    Distinction Test: factorization through existing parent → SUB-RISK

   

    Tests whether candidate factors through an existing L1 risk via

    attribute subsumption and impact concentration alignment.

   

    Args:

        candidate: The candidate to test

        l1_risks: List of potential parent L1 risks

       

    Returns:

        Tuple of (classification, parent_name or None)

    """

    parent = find_factorization_parent(candidate, l1_risks)

    if parent is not None:

        return Classification.SUB_RISK, parent.name

    return Classification.DRIVER, None

 

def run_materiality_test(candidate: Candidate, thresholds: Thresholds) -> Classification:

    """

    Materiality Test: HHI(c) ≥ threshold → SUB-RISK

   

    Tests whether impact distribution is concentrated (high HHI) or

    dispersed (low HHI, characteristic of cross-cutting drivers).

   

    Args:

        candidate: The candidate to test

        thresholds: Threshold configuration

       

    Returns:

        SUB-RISK if HHI meets threshold, DRIVER otherwise

    """

    hhi = compute_hhi(candidate)

    return Classification.SUB_RISK if hhi >= thresholds.hhi_threshold else Classification.DRIVER

 

def majority_rule(tests: List[Classification]) -> Classification:

    """

    Decision Rule: majority of tests (2/3 or 3/3) determines classification.

   

    Args:

        tests: List of individual test outcomes

       

    Returns:

        SUB-RISK if ≥2 tests indicate sub-risk

        DRIVER if ≥2 tests indicate driver

        UNDETERMINED if tie (should not occur with 3 tests)

    """

    sub_risk_count = sum(1 for t in tests if t == Classification.SUB_RISK)

    driver_count = sum(1 for t in tests if t == Classification.DRIVER)

   

    if sub_risk_count >= 2:

        return Classification.SUB_RISK

    elif driver_count >= 2:

        return Classification.DRIVER

    else:

        return Classification.UNDETERMINED

 

# =============================================================================

# SECTION 4: MAIN CLASSIFICATION PIPELINE

# =============================================================================

 

def classify_candidate(

    candidate: Candidate,

    l1_risks: List[L1Risk],

    thresholds: Thresholds = None

) -> ClassificationResult:

    """

    Execute the full classification pipeline for a single candidate.

   

    Steps:

    1. Compute metrics (BREADTH, CR, HHI)

    2. Run three classification tests

    3. Apply majority decision rule

    4. Return complete result with audit trail

   

    Args:

        candidate: The candidate concept to classify

        l1_risks: List of L1 risk categories in the taxonomy

        thresholds: Optional threshold configuration (uses defaults if None)

       

    Returns:

        ClassificationResult with metrics, test outcomes, and final classification

    """

    if thresholds is None:

        thresholds = Thresholds()

   

    # Step 1: Compute metrics

    breadth = compute_breadth(candidate)

    cr = compute_cr(candidate)

    hhi = compute_hhi(candidate)

   

    # Step 2: Run tests

    relevance = run_relevance_test(candidate, thresholds)

    distinction, parent_name = run_distinction_test(candidate, l1_risks)

    materiality = run_materiality_test(candidate, thresholds)

   

    # Step 3: Apply decision rule

    final = majority_rule([relevance, distinction, materiality])

   

    # Step 4: Build rationale for audit trail

    rationale_parts = [

        f"BREADTH={breadth} (affects {breadth} L1 categories)",

        f"CR={cr:.2f} ({'≥' if cr >= thresholds.cr_threshold else '<'} {thresholds.cr_threshold})",

        f"HHI={hhi:.2f} ({'≥' if hhi >= thresholds.hhi_threshold else '<'} {thresholds.hhi_threshold})",

        f"Tests: Relevance={relevance.value}, Distinction={distinction.value}, Materiality={materiality.value}",

    ]

    if parent_name:

        rationale_parts.append(f"Factorization parent: {parent_name}")

   

    return ClassificationResult(

        candidate_name=candidate.name,

        breadth=breadth,

        cr=cr,

        hhi=hhi,

        relevance_test=relevance,

        distinction_test=distinction,

        materiality_test=materiality,

        final_classification=final,

        parent_risk=parent_name,

        rationale="; ".join(rationale_parts)

    )

 

# =============================================================================

# SECTION 5: BATCH PROCESSING AND I/O

# =============================================================================

 

def load_input(filepath: str) -> Tuple[List[L1Risk], List[Candidate], Thresholds]:

    """

    Load L1 risks, candidates, and optional thresholds from JSON file.

   

    Args:

        filepath: Path to input JSON file

       

    Returns:

        Tuple of (l1_risks, candidates, thresholds)

    """

    with open(filepath, 'r', encoding='utf-8') as f:

        data = json.load(f)

   

    l1_risks = [

        L1Risk(

            name=r['name'],

            attributes=set(r.get('attributes', [])),

            definition=r.get('definition', '')

        )

        for r in data.get('l1_risks', [])

    ]

   

    candidates = [

        Candidate(

            name=c['name'],

            attributes=set(c['attributes']),

            impact_weights=c['impact_weights']

        )

        for c in data.get('candidates', [])

    ]

   

    thresholds = Thresholds()

    if 'thresholds' in data:

        thresholds.cr_threshold = data['thresholds'].get('cr', 0.50)

        thresholds.hhi_threshold = data['thresholds'].get('hhi', 0.25)

   

    return l1_risks, candidates, thresholds

 

def save_results(results: List[ClassificationResult], filepath: str):

    """

    Save classification results to JSON file.

   

    Args:

        results: List of classification results

        filepath: Output file path

    """

    output = {

        'classification_results': [

            {

                'candidate': r.candidate_name,

                'metrics': {

                    'breadth': r.breadth,

                    'cr': round(r.cr, 4),

                    'hhi': round(r.hhi, 4)

                },

                'tests': {

                    'relevance': r.relevance_test.value,

                    'distinction': r.distinction_test.value,

                    'materiality': r.materiality_test.value

                },

                'classification': r.final_classification.value,

                'parent_risk': r.parent_risk,

                'rationale': r.rationale

            }

            for r in results

        ]

    }

    with open(filepath, 'w', encoding='utf-8') as f:

        json.dump(output, f, indent=2)

 

def print_summary(results: List[ClassificationResult]):

    """Print a human-readable summary of classification results."""

    print(f"\nClassified {len(results)} candidates:\n")

    print("-" * 70)

    for r in results:

        status = "→ " + r.final_classification.value

        if r.parent_risk:

            status += f" (under {r.parent_risk})"

        print(f"  {r.candidate_name:25} {status}")

        print(f"    Metrics: BREADTH={r.breadth}, CR={r.cr:.2f}, HHI={r.hhi:.2f}")

        print(f"    Tests:   Relevance={r.relevance_test.value}, "

              f"Distinction={r.distinction_test.value}, "

              f"Materiality={r.materiality_test.value}")

        print()

    print("-" * 70)

 

def main():

    """Main entry point for command-line usage."""

    parser = argparse.ArgumentParser(

        description='Risk Taxonomy Classification Framework',

        epilog='See README.md for detailed documentation.'

    )

    parser.add_argument(

        '--input', '-i',

        required=True,

        help='Input JSON file with L1 risks and candidates'

    )

    parser.add_argument(

        '--output', '-o',

        required=True,

        help='Output JSON file for classification results'

    )

    parser.add_argument(

        '--verbose', '-v',

        action='store_true',

        help='Print detailed summary to console'

    )

    args = parser.parse_args()

   

    # Load data

    l1_risks, candidates, thresholds = load_input(args.input)

   

    # Classify all candidates

    results = [

        classify_candidate(c, l1_risks, thresholds)

        for c in candidates

    ]

   

    # Save results

    save_results(results, args.output)

   

    # Print summary

    if args.verbose:

        print_summary(results)

    else:

        print(f"Classified {len(results)} candidates:")

        for r in results:

            print(f"  {r.candidate_name}: {r.final_classification.value}")

   

    print(f"\nResults saved to: {args.output}")

 

if __name__ == '__main__':

    main()