# Spoon Estuary: TUFLOW FV-AED vs SCHISM-AED

This repository package contains a figure-led diagnostic comparison of the Spoon Estuary tutorial case using **TUFLOW FV-AED** and **SCHISM-AED**.

## View the report

- **Recommended:** publish the `docs/` folder with GitHub Pages.
- Local entry point: [`docs/index.html`](docs/index.html)
- Repository summary: [`REPORT.md`](REPORT.md)

The Pages layout uses equal-width model columns and fixed image stages. Tall composite figures were split into matched variable rows so that the two models are compared at a readable and consistent visual scale.

## Repository structure

```text
.
├── README.md
├── REPORT.md
├── UPLOAD_TO_GITHUB.md
├── docs/
│   ├── index.html
│   └── assets/
│       ├── crops/     # cropped panels used for matched-row comparisons
│       ├── figures/   # supplementary galleries
│       └── raw/       # original full-resolution figures
└── source/
    ├── model_output_plotting_tuflow.ipynb
    └── model_output_plotting_tuflow.pdf
```

## Publish with GitHub Pages

1. Upload this folder to the repository root.
2. Open **Settings → Pages**.
3. Choose **Deploy from a branch**.
4. Select the main branch and `/docs` folder.
5. Save.

## Comparison status

This is a diagnostic inter-model comparison, not a formal validation. Several variables still require unit harmonisation, and the current SCHISM-AED PAR/total-light outputs require configuration checking.

## Readability revision

This version uses a one-column layout for complex figures. Wide curtain plots, Hovmöller diagrams, profile/time-series figures and the GPP maps are no longer reduced into side-by-side thumbnails. TUFLOW Hovmöller and curtain crops were also regenerated to remove panel leakage and clipped labels.
