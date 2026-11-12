"""Load and merge the Lung3 clinical metadata and GEO gene-expression matrix.

Download the clinical spreadsheet from the TCIA NSCLC-Radiomics-Genomics
collection and the GSE58661 series matrix from GEO.
"""
from pathlib import Path
import argparse
import gzip
import pandas as pd


def read_geo_series_matrix(path):
    """Read GEO series-matrix data as samples x probes, indexed by sample title.

    GEO's column headers are usually GSM accessions, while the corresponding
    Lung3 clinical sheet uses patient/sample names. The !Sample_title metadata
    row provides the human-readable sample names used for the join.
    """
    import csv
    from io import StringIO

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        all_lines = handle.readlines()

    title_line = next(
        (line for line in all_lines if line.startswith("!Sample_title\t")),
        None
    )
    if title_line is None:
        raise ValueError("GEO file has no !Sample_title row to map samples.")

    titles = next(csv.reader([title_line], delimiter="\t", quotechar='"'))[1:]
    table_lines = [
        line for line in all_lines
        if not line.startswith("!") and line.strip()
    ]
    expression = pd.read_csv(StringIO("".join(table_lines)), sep="\t")
    if "ID_REF" not in expression.columns:
        raise ValueError("Could not find ID_REF in GEO series matrix.")

    gsm_columns = expression.columns[1:].tolist()
    if len(titles) != len(gsm_columns):
        raise ValueError(
            "Number of GEO sample titles does not match expression columns."
        )

    expression = expression.set_index("ID_REF").T
    expression.index = pd.Index(titles, name="sample_id")
    return expression


def normalize_id(values):
    """Normalize Lung sample identifiers for joining."""
    return (
        pd.Series(values, dtype="string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )


def load_clinical(path, id_column, target_column):
    """Load the TCIA clinical spreadsheet and validate required columns."""
    clinical = pd.read_excel(path)
    clinical.columns = [str(c).strip() for c in clinical.columns]
    if id_column not in clinical.columns:
        raise KeyError(
            f"ID column {id_column!r} not found. Available columns: "
            f"{clinical.columns.tolist()}"
        )
    if target_column not in clinical.columns:
        raise KeyError(
            f"Target column {target_column!r} not found. Available columns: "
            f"{clinical.columns.tolist()}"
        )
    clinical = clinical.copy()
    clinical["sample_id"] = normalize_id(clinical[id_column]).values
    clinical = clinical.drop(columns=[id_column])
    return clinical


def build_dataset(clinical_path, expression_path, id_column, target_column):
    """Merge clinical and expression data by patient/sample identifier."""
    clinical = load_clinical(clinical_path, id_column, target_column)
    expression = read_geo_series_matrix(expression_path)

    # GEO sample titles are used as sample identifiers, rather than GSM IDs.
    expression = expression.copy()
    expression.index = normalize_id(expression.index).values

    # If GEO sample identifiers are GSM accessions, use the GEO sample
    # annotation/title mapping to map them to the matching Lung patient IDs.
    common = clinical["sample_id"].isin(expression.index)
    if not common.any():
        raise ValueError(
            "No matching patient IDs found between clinical and expression "
            "data. Inspect GEO sample titles and TCIA patient identifiers, "
            "then create an explicit sample-ID mapping before merging."
        )

    clinical = clinical.loc[common].set_index("sample_id")
    expression = expression.loc[expression.index.intersection(clinical.index)]
    merged = clinical.join(expression, how="inner", rsuffix="_gene")
    merged = merged.loc[merged[target_column].notna()]
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", default="data/Lung3.metadata.xls")
    parser.add_argument("--expression", default="data/GSE58661_series_matrix.txt.gz")
    parser.add_argument("--id-column", required=True,
                        help="Exact patient/sample identifier column in the XLS file.")
    parser.add_argument("--target", default="Histology")
    parser.add_argument("--output", default="data/merged_dataset.csv")
    args = parser.parse_args()

    data = build_dataset(
        args.clinical, args.expression, args.id_column, args.target
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=True)
    print(f"Saved {data.shape[0]} samples and {data.shape[1] - 1} columns to {args.output}")
    print("Target distribution:")
    print(data[args.target].value_counts(dropna=False))


if __name__ == "__main__":
    main()
