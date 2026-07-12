# Spoon Estuary boundary audit

## Confirmed geometry

- TUFLOW nodestring ID 1 is the lower/southern end of the model and is the river Q boundary.
- TUFLOW nodestring ID 2 is the upper/northern end and is the ocean water-level boundary.
- Current SCHISM ocean open boundary is at the upper/northern end, using nodes:
  1402, 1408, 1413, 1417, 1419, 1418, 1415.
- Current SCHISM river input is not an open boundary. It is a 20 m3/s point/element source in element 1 at the lower/southern end.
- The SCHISM mesh nodes corresponding to the TUFLOW river cross-section are:
  15, 10, 6, 3, 1.
  These five nodes are currently included in the exterior land boundary.

## Forcing comparison

- River discharge time series: exact numerical match, 20 m3/s for all 744 hourly records.
- River temperature, salinity and all 17 AED values: exact numerical match.
- Ocean water level, temperature, salinity and all 17 AED values: exact numerical match.
- Time steps and record times: exact match for the supplied 31-day forcing files.

## Critical mismatch

The boundary values are copied correctly, but the river hydrodynamic boundary implementation is not equivalent:

- TUFLOW: distributed Q boundary across nodestring ID 1.
- Current SCHISM: all 20 m3/s injected into source element 1 through source_sink.in/vsource.th.
- The actual lower cross-section remains classified as land in hgrid.gr3/hgrid.ll.
- SCHISM dramp_ss = 1 day also ramps the source during the first day, while the TUFLOW Q series is 20 m3/s from the first record.

Therefore the current setup is spatially located at the correct end, but it is not a strict boundary-for-boundary reproduction.

## Required correction for strict comparison

1. Convert SCHISM nodes 15, 10, 6, 3, 1 into a second open-boundary segment.
2. Prescribe river discharge as -20 m3/s in SCHISM; negative means inflow.
3. Apply the river temperature, salinity and 17 AED concentrations at that river open boundary.
4. Keep the existing seven-node ocean boundary for water level and ocean constituent concentrations.
5. Remove the element-source representation and disable source ramping for this comparison.
6. Split the exterior land boundary around the new river opening.
