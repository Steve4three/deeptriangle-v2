# Data

This repository does **not** include the Schedule P dataset used in the paper.

The experiments expect two CSV files in this directory:

- `triangle_sample.csv`
- `triangle_company_info.csv`

These files are derived from a proprietary NAIC Schedule P compilation and are **not redistributable**.

To reproduce results with public data, adapt the loader in `data_prep.py` to the CAS Loss Reserve Database format and regenerate these two CSVs.
