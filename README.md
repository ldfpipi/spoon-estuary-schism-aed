# Spoon Estuary SCHISM-AED

## Overview

This repository presents the current SCHISM-AED setup and preliminary
hydrodynamic and water-quality results for the Spoon Estuary multi-model
benchmark.

The SCHISM-AED case has been successfully set up and run on UQ Bunya.
The current purpose of this repository is to present the available model
setup, boundary audit and preliminary results for internal review and
feedback from the project team.

## Current status

- SCHISM-AED compiled and run successfully on UQ Bunya
- Stable hydrodynamic, tracer and AED model execution achieved
- Hourly hydrodynamic and water-quality outputs generated
- River and ocean boundary locations audited
- River discharge, ocean water level, temperature, salinity and AED
  forcing values checked against the TUFLOW FV-AED reference
- Preliminary result figures produced following the broad structure of
  Matt's TUFLOW plotting notebook

## Setup comparison

Both the TUFLOW FV-AED and SCHISM-AED cases represent:

- one river freshwater input
- one ocean/tidal boundary
- temperature and salinity transport
- 17 active AED state variables
- the same overall Spoon Estuary benchmark domain

The river and ocean ends are consistently located in both models:

- the river is at the lower/southern end
- the ocean/tidal boundary is at the upper/northern end

The main current implementation difference is that TUFLOW applies the
river inflow as a distributed discharge boundary, whereas the current
SCHISM-AED case applies the same 20 m³/s river inflow through a local
element source.

The current results should therefore be viewed as a successful setup and
results demonstration for team discussion, rather than a final formal
one-to-one model assessment.

## Preliminary results

The current result set includes:

- water-surface elevation
- dissolved oxygen
- total nitrogen
- gross primary production
- velocity
- total suspended solids
- total phosphorus
- surface, near-seabed and depth-averaged fields

## Report

The latest report will be available in the `reports/` folder in both PDF
and Word formats.

## Repository structure

```text
spoon-estuary-schism-aed/
├── README.md
├── reports/
├── figures/
├── notebooks/
├── model_inputs/
├── boundary_audit/
└── scripts/
