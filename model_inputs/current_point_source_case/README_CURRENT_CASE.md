# Current point-source case - not the final strict benchmark

This folder contains the audited SCHISM input subset used for the present results.

Critical limitation: river discharge is injected into element 1 through `source_sink.in`, `vsource.th` and `msource.th`. The corresponding lower cross-section nodes `15, 10, 6, 3, 1` remain classified as land boundary nodes.

Before formal TUFLOW-SCHISM comparison, create a separate strict river open-boundary case and retain this folder only for traceability.
