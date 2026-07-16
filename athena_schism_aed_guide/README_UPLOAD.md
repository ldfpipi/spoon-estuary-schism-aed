# Upload this page to the existing GitHub Pages repository

Target repository:
https://github.com/ldfpipi/spoon-estuary-schism-aed

Recommended folder name:
athena_schism_aed_guide

Files in this folder:
- index.html
- run_spoon_sczaed_6day_athena.pbs
- SCHISM-AED_Compilation_on_Athena.pdf

Browser upload procedure:
1. Open the repository.
2. Select **Add file > Upload files**.
3. Drag the entire `athena_schism_aed_guide` folder into the upload area.
4. Use the commit message:
   `Add Athena SCHISM-AED installation and runtime guide`
5. Commit directly to `main`.
6. Wait 1-3 minutes for GitHub Pages to refresh.
7. Open:
   https://ldfpipi.github.io/spoon-estuary-schism-aed/athena_schism_aed_guide/index.html

If GitHub flattens the folder during browser upload:
1. Create the folder `athena_schism_aed_guide` in the repository first.
2. Open that folder.
3. Use **Add file > Upload files** and upload the three files.

The repository must already have GitHub Pages enabled. If it does not:
Settings > Pages > Build and deployment > Deploy from a branch > main > /(root) > Save.

Validated runtime request: 64 MPI ranks and 20 GB memory.
