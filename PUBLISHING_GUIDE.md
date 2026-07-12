# Publishing guide

## Recommended route

Add this folder to the existing `AquaticEcoDynamics/spoon_estuary_multi_model` repository through a reviewed branch, fork or pull request. This keeps TUFLOW FV-AED and SCHISM-AED under one project link.

Suggested destination:

```text
spoon_estuary_multi_model/spoon_sczaed_bunya/
```

## Before publishing

1. Confirm permission from the repository maintainers and project leads.
2. Confirm that the model inputs, selected outputs and acknowledgements may be public.
3. Agree on a software/data licence.
4. Remove personal paths, credentials, binaries and restricted data.
5. Preserve the label `current_point_source_case` until the strict river open-boundary case is complete.

## Example Git workflow

```bash
git clone https://github.com/AquaticEcoDynamics/spoon_estuary_multi_model.git
cd spoon_estuary_multi_model
git checkout -b add-bunya-schism-results
cp -r /path/to/spoon_sczaed_bunya_publish_package spoon_sczaed_bunya
git add spoon_sczaed_bunya
git commit -m "Add Bunya SCHISM-AED setup, boundary audit and results gallery"
git push -u origin add-bunya-schism-results
```

Then open a pull request for team review.

## Large outputs

Do not commit full SCHISM output folders or large NetCDF files to normal Git history. Use an approved data archive and link it from the README. Git LFS is an option for selected large files, but a scientific data repository is preferred for complete model datasets.

## Optional GitHub Pages

The `docs/index.md` file is a ready starting point for a project website. Enable GitHub Pages from the repository settings after the folder is merged and reviewed.
