# Upload checklist

```bash
git clone <your-repository-url>
cd <your-repository>
cp -r /path/to/spoon_estuary_github_report_balanced/* .
git add README.md REPORT.md UPLOAD_TO_GITHUB.md docs source
git commit -m "Add balanced TUFLOW FV-AED and SCHISM-AED comparison report"
git push
```

Then open **Settings → Pages**, choose **Deploy from a branch**, and select the main branch with the `/docs` folder.
