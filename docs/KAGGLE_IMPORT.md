# Importing Kaggle-Trained Models to Windows (or any local machine)

This guide explains how to export the model registry from a Kaggle training run
and import it into your local Windows (or any OS) clone of this repository so
the Streamlit webapp shows all trained models.

---

## Background

When you train on Kaggle, checkpoints are saved to:

```
/kaggle/working/MedRAG/models/registry/checkpoints/<version>.pt
```

and metadata is stored in:

```
/kaggle/working/MedRAG/models/registry/registry.json
```

Because these files live on Kaggle's ephemeral disk they are not pushed to
GitHub.  The import script in this repo bridges the gap: it copies the
artifacts into your local `models/registry/` directory **and** rewrites all
absolute Kaggle paths to portable repo-relative paths.

---

## Step 1 – Export from Kaggle

Run the following cell in your Kaggle notebook **after training completes**:

```python
import shutil, os
os.chdir('/kaggle/working/MedRAG')
shutil.make_archive('/kaggle/working/registry_export', 'zip', '.', 'models/registry')
print("Created /kaggle/working/registry_export.zip")
```

Then download `registry_export.zip` from the Kaggle output panel.

---

## Step 2 – Import on Windows (or Linux/macOS)

1. Clone / pull the repository (if you haven't already):

   ```bash
   git clone <your-repo-url>
   cd MedRAG
   ```

2. Place the downloaded `registry_export.zip` anywhere on your machine (e.g.
   your `Downloads` folder).

3. Run the import script, pointing it at the zip:

   ```bash
   python scripts/import_registry_artifacts.py C:\Users\YourName\Downloads\registry_export.zip
   ```

   Or, if you extracted the zip first, pass the folder:

   ```bash
   python scripts/import_registry_artifacts.py C:\Users\YourName\Downloads\models\registry
   ```

4. The script will:
   - Copy all `*.pt` checkpoint files into `models/registry/checkpoints/`
   - Merge the imported entries into `models/registry/registry.json`
   - Rewrite every `checkpoint_path` value from
     `/kaggle/working/MedRAG/models/registry/checkpoints/<name>.pt`
     to the portable `models/registry/checkpoints/<name>.pt`
   - Print a summary of how many versions were imported and how many
     checkpoint files are missing (if any)

Example output:

```
Extracting 'registry_export.zip' …
Found registry at: models/registry
  COPIED  v1.0_round1_20260402_082246.pt
  COPIED  v1.1_round1_20260402_082246.pt
  ...
=======================================================
  Import complete
  Versions imported : 50
  Missing checkpoints: 0
  Registry saved to : /path/to/MedRAG/models/registry/registry.json
=======================================================
```

---

## Step 3 – Launch the Streamlit Webapp

```bash
streamlit run webapp/Home.py
```

Navigate to **Model Registry** in the sidebar.  You should now see all 50
(or however many) trained versions with their metrics.

> **Note**: If a checkpoint file is missing the webapp will still list the
> version with a warning instead of crashing.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Registry shows 0 models | `registry.json` still has Kaggle paths | Re-run the import script |
| "Checkpoint not found" warning | `.pt` files were not included in the zip | Re-export on Kaggle including `checkpoints/` |
| Import script says "Cannot find registry.json" | Zip does not have the expected layout | Ensure the zip was created with `models/registry` as the root |

---

## Notes

- **`.pt` files are excluded from git** (they can be hundreds of MB each).
  Only `registry.json` and `.gitkeep` placeholder files are committed.
- If you retrain on Kaggle and want to update your local registry, simply
  run the import script again – it **merges** new versions rather than
  replacing the existing registry.
