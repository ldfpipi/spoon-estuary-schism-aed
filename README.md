# Spoon Estuary SCHISM-AED on UQ Bunya

This folder documents the Bunya-ready SCHISM-AED implementation of the Spoon Estuary benchmark and presents preliminary hydrodynamic and water-quality results in figure groups aligned with the TUFLOW FV-AED plotting notebook.

> **Current status:** the river and ocean ends and all supplied boundary values have been verified. The present SCHISM river inflow is still represented by a source in element 1, whereas TUFLOW uses a distributed Q boundary. Treat the included inputs and figures as the **current point-source case**, not the final strict intercomparison case.

## Start here

- [Setup and results report](docs/setup_and_results_report.md)
- [Executed results gallery notebook](notebooks/SCHISM_AED_results_gallery.ipynb)
- [Boundary audit](boundary_audit/boundary_audit_report.md)
- [Publication guide](PUBLISHING_GUIDE.md)

## Headline findings

- TUFLOW river boundary: lower/southern nodestring ID 1, Q = 20 m³/s.
- TUFLOW ocean boundary: upper/northern nodestring ID 2, prescribed water level.
- Current SCHISM ocean boundary: matching upper end, 7 open-boundary nodes.
- Current SCHISM river input: matching lower end, but injected into source element 1.
- River/ocean discharge, water level, temperature, salinity and all 17 AED time series match numerically in the audited files.
- Required correction: create a 5-node SCHISM river open boundary at nodes 15, 10, 6, 3 and 1; remove the element source and unmatched ramping.

## Results gallery

![Oxygen, TN and GPP](figures/oxygen_tn_gpp.png)

![Velocity, TSS and TP](figures/velocity_tss_tp_3x3.png)

## Data availability

Full SCHISM NetCDF output is intentionally excluded from normal Git history. Store the complete output dataset in an approved research-data repository and add its DOI or access link here after project approval.

## Contributors and review

Prepared by Dongfang Liu. The benchmark should be reviewed with Matt Hipsey, Badin Gibbes and the Aquatic EcoDynamics/Spoon Estuary project team before public release.
