# ==============================================================================
# run_pipeline.py
# Run notebooks in the right order with logs and chronomèter
# Usage : python run_pipeline.py
# ==============================================================================

import subprocess
import time
import sys
from datetime import datetime

# ==============================================================================
# CONFIG — notebook list in order of execution
# ==============================================================================

NOTEBOOKS = [
    # "1_code_find_and_merge_xl.ipynb",
    # "2_merging_cleaning_vital_signs_and_ioa_file.ipynb",
    # "3_merging_ioaparamfile_med_adm.ipynb",
    # "4_radio_cleaning_and_merge.ipynb",
    # "5_bio_cleaning_and_merge.ipynb",
    #"6a_compare_hospit_uhcd_2022.ipynb",
    "6b_full_df_tabular_cleaning_harmonization_imputation.ipynb",
    "7_descriptive_analysis-clean.ipynb",
    "8_correlation_hospit_vs_conso.ipynb",
    "9_clustering_pipeline.ipynb",
    #"9bis_BERTopic.ipynb"
    "10_clusters_and_outliers_description.ipynb",
    "10_decision_trees.ipynb",
    "12_logistic_regression,ipynb"

]

LOG_FILE = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ==============================================================================
# HELPERS
# ==============================================================================

def log(msg: str):
    """Print and write to log file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line      = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_notebook(notebook: str) -> bool:
    """
    Execute a notebook using nbconvert.
    Returns True if success, False if error.
    """
    log(f"▶ Starting : {notebook}")
    t_start = time.time()

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=86400",  # 24h max
            notebook,
        ],
        capture_output = True,
        text           = True,
    )

    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode == 0:
        log(f"✅ Done    : {notebook} — {minutes}m {seconds}s")
        return True
    else:
        log(f"❌ FAILED  : {notebook} — {minutes}m {seconds}s")
        log(f"   Error   : {result.stderr[-500:]}")  # dernières 500 chars de l'erreur
        return False


# ==============================================================================
# PIPELINE
# ==============================================================================

def run_pipeline():
    log("=" * 60)
    log(f"PIPELINE START — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Notebooks : {len(NOTEBOOKS)}")
    log(f"Log file  : {LOG_FILE}")
    log("=" * 60)

    t_total   = time.time()
    n_success = 0
    n_failed  = 0
    failed    = []

    for i, notebook in enumerate(NOTEBOOKS, 1):
        log(f"\n[{i}/{len(NOTEBOOKS)}] ─────────────────────────────────────")
        success = run_notebook(notebook)

        if success:
            n_success += 1
        else:
            n_failed += 1
            failed.append(notebook)

            # Arrêter si un notebook échoue ?
            # Commenter les 3 lignes suivantes pour continuer malgré l'erreur
            log("Pipeline stopped due to error.")
            log("To continue despite errors, comment the 'break' line.")
            break

    # ── Résumé ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    log("\n" + "=" * 60)
    log(f"PIPELINE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total time : {hours}h {minutes}m {seconds}s")
    log(f"Success    : {n_success} / {len(NOTEBOOKS)}")
    log(f"Failed     : {n_failed} / {len(NOTEBOOKS)}")
    if failed:
        log(f"Failed notebooks :")
        for nb in failed:
            log(f"  ❌ {nb}")
    log("=" * 60)


if __name__ == "__main__":
    run_pipeline()


