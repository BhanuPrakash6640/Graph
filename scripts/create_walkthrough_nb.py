"""Generator script for explainability walkthrough notebook."""
import json
from pathlib import Path


def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GraphGuard: Explainability Walkthrough\n",
                    "\n",
                    "**Research Demonstration on Historical Elliptic Bitcoin Dataset (Weber et al. 2019)**\n",
                    "\n",
                    "> **Disclaimer:** Features local_1..93 and agg_1..72 are strictly anonymised. This system is an offline research demonstration and does not monitor live blockchain activity or make regulatory compliance determinations.\n",
                    "\n",
                    "This walkthrough analyzes three test transactions flagged by GraphGuard:\n",
                    "1. **True Positive 1 (`txId: 69746416`)**: Step 35 illicit transaction with extreme high-risk feature attributions.\n",
                    "2. **True Positive 2 (`txId: 30585812`)**: Step 37 illicit transaction embedded in a multi-hop laundering pattern.\n",
                    "3. **False Positive (`txId: 106364923`)**: Step 44 licit transaction flagged as illicit due to chain layering and proximity to high-risk transactions during the darknet market shutdown regime shift.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "\n",
                    "explanations_dir = Path('../../data/processed/explanations/xgb_gcn_paper_refit_all')\n",
                    "if not explanations_dir.exists():\n",
                    "    explanations_dir = Path('data/processed/explanations/xgb_gcn_paper_refit_all')\n",
                    "\n",
                    "def load_tx(tx_id):\n",
                    "    with open(explanations_dir / f'{tx_id}.json') as f:\n",
                    "        return json.load(f)\n",
                    "\n",
                    "tp1 = load_tx('69746416')\n",
                    "tp2 = load_tx('30585812')\n",
                    "fp = load_tx('106364923')\n",
                    "\n",
                    "print(f\"Loaded TP1 ({tp1['txId']}): score={tp1['score']}, label={tp1['known_label']}\")\n",
                    "print(f\"Loaded TP2 ({tp2['txId']}): score={tp2['score']}, label={tp2['known_label']}\")\n",
                    "print(f\"Loaded FP ({fp['txId']}): score={fp['score']}, label={fp['known_label']}\")\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. True Positive Analysis: txId 69746416 (Time Step 35)\n",
                    "\n",
                    "This transaction scored **1.000** against threshold **0.4656**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_tp1 = pd.DataFrame(tp1['top_features'])\n",
                    "print(\"Top SHAP Feature Attributions for TP1:\")\n",
                    "print(df_tp1.to_string(index=False))\n",
                    "print(\"\\nPattern Flags:\", tp1['pattern_flags'])\n",
                    "print(\"Fidelity:\", tp1['fidelity'])\n",
                    "print(\"Top 3 Neighbor Contributions:\", tp1['neighbor_contributions'][:3])\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. True Positive Analysis: txId 30585812 (Time Step 37)\n",
                    "\n",
                    "This transaction scored **1.000** against threshold **0.4656**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_tp2 = pd.DataFrame(tp2['top_features'])\n",
                    "print(\"Top SHAP Feature Attributions for TP2:\")\n",
                    "print(df_tp2.to_string(index=False))\n",
                    "print(\"\\nPattern Flags:\", tp2['pattern_flags'])\n",
                    "print(\"Fidelity:\", tp2['fidelity'])\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. False Positive Analysis: txId 106364923 (Time Step 44)\n",
                    "\n",
                    "**Ground Truth:** Licit (`y = 0`)\n",
                    "**Model Prediction:** Illicit (`score = 0.9960 >= 0.4656`)\n",
                    "\n",
                    "### Why did the model falsely flag this transaction?\n",
                    "1. **Covariate Shift in Step 44:** In time step 44 (the onset of the darknet market shutdown), overall network activity and transaction structures changed drastically.\n",
                    "2. **Feature Profile:** Features `local_53` (+2.08 SHAP) and `local_90` (+1.41 SHAP) took values identical to the illicit clusters learned during steps 1–34.\n",
                    "3. **Graph Topology:** The transaction exhibited a `chain_node` pattern (layering / pass-through) with `high_illicit_neighborhood` (neighbors also scored high risk). This combination misled both the tree-based and graph-based models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_fp = pd.DataFrame(fp['top_features'])\n",
                    "print(\"False Positive Feature Attribution Profile:\")\n",
                    "print(df_fp.to_string(index=False))\n",
                    "print(\"\\nPattern Flags:\", fp['pattern_flags'])\n",
                    "print(\"Fidelity:\", fp['fidelity'])\n",
                    "print(\"Neighbor Occlusion Deltas:\", fp['neighbor_contributions'])\n"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.12"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    nb_path = Path("experiments/notebooks/01_explainability_walkthrough.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook at {nb_path}")


if __name__ == "__main__":
    create_notebook()

