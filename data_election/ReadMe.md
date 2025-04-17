# Data Election Dataset

This repository contains digitized election ballot forms (C1 forms) from the Indonesian election system (SIREKAP), organized for digit recognition and analysis tasks.

## Dataset Overview

| Component | Description |
|-----------|-------------|
| `batch_inference.csv` | Labels and metadata for all digit images |
| `digits/` | Main directory containing all digit images organized by ballot ID |
| `json/` | JSON files containing annotations for each ballot |
| `json_batch_norm_2_tent/` | JSON files processed with BatchNorm2 and TENT |
| `json_batch_norm_3_tent_real/` | JSON files processed with BatchNorm3 and TENT |

## Image Organization

Each ballot has a unique ID (e.g., `1101012015002`) with the following structure:

| Level | Description |
|-------|-------------|
| Ballot Level | `digits/[ballot_id]/` |
| Candidate Level | `candidate1/`, `candidate2/`, `candidate3/` |
| Digit Level | `digit1/` (hundreds), `digit2/` (tens), `digit3/` (ones) |

**Note**: Not all candidates have all digit positions. For example, if a candidate received fewer than 100 votes, the `digit1` folder may be absent.

## Labels and Annotations

- `batch_inference.csv` contains the ground truth labels (0-9) for each digit image
- JSON files in the `json/` directory provide additional metadata for each ballot
- The two JSON subdirectories contain processed annotations from different experiments

This dataset was used in experiments comparing different preprocessing methods and test-time adaptation techniques (TENT) with various batch normalization configurations.
