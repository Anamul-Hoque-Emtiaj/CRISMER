# CRISMER-opti

CRISMER-opti is a tool for CRISPR guide RNA design and optimization.

## Dependencies

### Python Packages
- pandas 
- numpy
- torch 
- scikit-learn 
- scipy 
- joblib

### External Tools
- Cas-OFFinder==2.4 - For off-target search in genomes

### Data Requirements
- Reference genome file (Can be downloaded from the UCSC website, hg38 is recommended)
- Pre-trained models (included in models/ directory)

## Set Up

1. Create a python3 environment (conda is recommended)

2. Install required python libraries
```bash
# Using conda
conda install pandas numpy scipy joblib scikit-learn
conda install pytorch -c pytorch

# Alternative: using pip
pip install pandas numpy torch scikit-learn scipy joblib
```

3. Download and install Cas-OFFinder. NOTE: Properly set the environment variables to make Cas-OFFinder available.
   - Visit: https://github.com/snugel/cas-offinder for installation instructions

4. Download genome reference file from the UCSC website and uncompress it to data/. (e.g., data/hg38.fa).
   - Visit: http://www.rgenome.net/cas-offinder/portable

## Usage

The tool includes several modules for CRISPR guide RNA analysis:

- **CRISMER-Opti**: Optimize sgRNAs through strategic mutations to improve specificity while maintaining high activity. This is the primary functionality of the tool.
- CRISMER off_spec: Calculate the genome wide specificity of sgRNAs
- CRISMER-Score: Calculate the activity score of sgRNA-DNA pairs
- CRISMER-Spec: Calculate the specificity of given sgRNA



## Example

```bash
# Optimize a target sequence for improved specificity
python CRISMER.py opti --tar CGTGCGCAGGAGGACGAGGACGG --genome hg38.fa --out example/pcsk9_out.csv

# Genome wide specificity
python CRISMER.py off_spec --sgr GAGTCCGAGCAGAAGAAGAANGG --tar GAGTCCGAGCAGAAGAAGAAGGG --genome data/hg38.fa
# Calculate CRISMER-Score for a sgRNA-target pair
python CRISMER.py score --sgr ACGTACGTACGTACGTACGTAGG --tar ACGTACGTACGTACGTACGTAGG

# Calculate scores for multiple sgRNA-target pairs
python CRISMER.py scores --csv example/example.csv


```
