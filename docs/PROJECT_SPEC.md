# PROJECT SPECIFICATION: GraphGuard
## Temporal Graph Intelligence for Financial Crime Detection

Act as a Senior Machine Learning Research Engineer, Graph ML Specialist, and Full-Stack AI Architect.

I want to build a research-oriented, production-quality machine learning project called **GraphGuard — Temporal Graph Intelligence for Financial Crime Detection**.

This project must demonstrate genuine ML engineering, graph neural networks, temporal learning, rigorous experimentation, explainable AI, and production deployment.

This is NOT a basic fraud detection project. Do not build a simple Random Forest classifier with a dashboard and call it an AI product.

The objective is to create something technically impressive enough to showcase to recruiters at companies such as Google, Microsoft, NVIDIA, Amazon, JPMorgan, and leading AI/fintech startups.

---

# 1. PROJECT VISION

Traditional fraud detection systems generally analyze transactions individually.

However, financial crime often involves networks of interconnected transactions.

Consider this example:

- Wallet A sends money to Wallet B.
- Wallet B transfers money to Wallet C.
- Wallet C distributes money across several other wallets.
- Individually, these transactions might appear legitimate.
- Collectively, the transaction network might reveal suspicious behavior.

GraphGuard will use Graph Neural Networks (GNNs) and temporal graph learning to identify suspicious transaction patterns that conventional machine learning models may overlook.

### Core research question

"Does incorporating transaction relationships and temporal information improve the detection of illicit financial activity compared with conventional machine learning approaches?"

The project must investigate this question experimentally rather than assume that GNNs always outperform traditional models.

# 2. DATASET

Use the Elliptic Bitcoin Transaction Dataset.

Official dataset:
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Research paper:
https://arxiv.org/abs/1908.02591

Dataset characteristics:

- 203,769 transaction nodes.
- 234,355 directed transaction edges.
- 166 transaction features.
- 49 time intervals.
- Three label categories:
  - Illicit
  - Licit
  - Unknown

Use this dataset as the primary benchmark.

Important:
- Unknown labels must not automatically be treated as legitimate.
- Understand the dataset's temporal structure.
- Prevent target leakage.
- Preserve the original train/test protocol where appropriate.
- Also implement a stricter chronological evaluation protocol.
- Document every preprocessing decision.

# 3. MACHINE LEARNING ARCHITECTURE

Implement and compare multiple approaches.

## A. Traditional ML baselines

Implement:

1. Logistic Regression
2. Random Forest
3. XGBoost

These models establish the baseline against which graph-based methods will be evaluated.

Use transaction-level features.

## B. Graph Neural Networks

Implement the following architectures:

### 1. Graph Convolutional Network (GCN)

Purpose:
Learn node representations by aggregating information from neighboring transaction nodes.

### 2. GraphSAGE

Purpose:
Learn inductive node representations through neighborhood sampling and aggregation.

Experiment with:
- Mean aggregation.
- Max pooling aggregation where practical.
- Different neighborhood sampling sizes.

### 3. Graph Attention Network (GAT)

Purpose:
Use attention mechanisms to learn the importance of neighboring transactions.

Investigate whether attention weights provide useful insights into suspicious transaction relationships.

## C. Temporal Graph Learning

This is the advanced component.

Investigate how transaction behavior evolves over time.

Implement a temporal approach that respects the chronological ordering of transactions.

Potential approaches:

- Temporal Graph Networks (TGN).
- Time-aware GraphSAGE.
- Temporal feature aggregation.

Do not implement every advanced architecture just for the sake of adding names. Prioritize one working temporal model with defensible evaluation.

# 4. EXPERIMENTAL DESIGN

This is one of the most important parts of the project.

Create a proper experimental framework.

### Experiment 1: Traditional ML vs GNN

Compare:
- Logistic Regression
- Random Forest
- XGBoost
- GCN
- GraphSAGE
- GAT

Question:
Does graph information improve illicit transaction detection?

### Experiment 2: Temporal vs Non-temporal Learning

Compare:
- Static GNN
- Temporal GNN

Question:
Does incorporating chronological information improve detection?

### Experiment 3: Generalization

Test models on future time periods that were not included in training.

Question:
Can the model detect suspicious transactions under temporal distribution shifts?

### Experiment 4: Class Imbalance

Investigate:
- Class weighting.
- Focal loss.
- Oversampling where appropriate.
- Threshold optimization.

Avoid misleading accuracy metrics.

### Experiment 5: Ablation Study

Evaluate the contribution of:

- Transaction features.
- Graph topology.
- Neighbor information.
- Temporal information.

Determine which components actually improve performance.

# 5. EVALUATION METRICS

Do not rely on accuracy alone.

Track:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- False positive rate
- Inference latency
- Training time
- GPU/CPU memory consumption

PR-AUC is particularly important because illicit transactions are a minority class.

Create comparison tables and visualizations for every experiment.

Never fabricate results or benchmark numbers.

# 6. EXPLAINABLE AI

A fraud detection system must provide understandable explanations.

Implement an explainability layer that helps investigators understand why a transaction was flagged.

Potential techniques:

- GNNExplainer.
- Feature importance analysis.
- SHAP for compatible traditional ML models.
- Neighbor contribution analysis.
- Graph substructure visualization.

For each flagged transaction, display:

1. Predicted risk score.
2. Predicted class.
3. Important transaction features.
4. Relevant neighboring nodes.
5. Suspicious graph patterns.
6. Model explanation.

Do not present model attention weights as definitive causal explanations.

# 7. PRODUCT: GRAPHGUARD DASHBOARD

Build a polished web application.

The interface should resemble a professional fintech risk intelligence platform.

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- Recharts
- Cytoscape.js or React Flow for graph visualization

### Backend

- Python
- FastAPI
- PyTorch
- PyTorch Geometric
- Scikit-learn
- XGBoost

### Storage

Use a sensible architecture:
- SQLite or PostgreSQL for application metadata.
- Efficient storage for processed graph data.
- Model checkpoints stored separately.

Do not unnecessarily introduce complex infrastructure.

# 8. DASHBOARD MODULES

Build the following pages.

### A. Overview Dashboard

Display:
- Total transactions analyzed.
- Illicit transactions identified in labeled evaluation data.
- Model performance.
- Detection trends.
- Recent inference results.
- Model comparison.

Clearly distinguish historical dataset statistics from live predictions.

### B. Transaction Investigation

Allow users to:
- Enter a transaction ID.
- Retrieve its features.
- Run inference.
- Display predicted risk.
- Visualize its local transaction neighborhood.
- View model explanations.

### C. Graph Explorer

Create an interactive transaction graph.

Features:
- Nodes represent transactions.
- Edges represent transaction relationships.
- Different node colors represent predicted or known classes.
- Click a node to inspect its details.
- Highlight neighboring nodes.
- Highlight suspicious subgraphs.

Ensure that unknown labels are visually distinct from legitimate labels.

### D. Model Benchmark

Display:
- Traditional ML vs GNN performance.
- Model comparison charts.
- PR curves.
- ROC curves.
- Confusion matrices.
- Training time.
- Inference latency.

### E. Temporal Analysis

Display:
- Transaction activity over time.
- Detection trends across time intervals.
- Temporal performance.
- Distribution shifts.
- Model generalization.

### F. Explainability

Display:
- Feature contributions.
- Important neighboring transactions.
- Graph explanations.
- Local subgraph visualizations.

# 9. API DESIGN

Implement REST APIs using FastAPI.

Suggested endpoints:

GET /api/health

GET /api/models

POST /api/predict

GET /api/transaction/{transaction_id}

GET /api/transaction/{transaction_id}/neighbors

GET /api/transaction/{transaction_id}/explanation

GET /api/benchmark

GET /api/temporal-analysis

POST /api/experiments/run

Include:
- Request validation.
- Response schemas.
- Error handling.
- Model loading and caching.
- API documentation.
- Proper logging.

Do not load the entire model repeatedly for every request.

# 10. RESEARCH AND DOCUMENTATION

Create a proper research-style technical report.

Suggested structure:

1. Abstract
2. Introduction
3. Problem Statement
4. Related Work
5. Dataset Description
6. Methodology
7. Model Architectures
8. Experimental Setup
9. Results
10. Ablation Studies
11. Explainability
12. Limitations
13. Future Work
14. Conclusion
15. References

Include:
- Architecture diagrams.
- Dataset visualizations.
- Model comparison tables.
- Training curves.
- PR curves.
- Confusion matrices.
- Ablation study results.

The report must reflect actual experiments.

Do not invent research results, citations, or claims of novelty.

# 11. ENGINEERING QUALITY

The GitHub repository should look like a serious research engineering project.

Suggested structure:

GraphGuard/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   └── lib/
│
├── ml/
│   ├── data/
│   ├── preprocessing/
│   ├── baselines/
│   ├── gnn/
│   ├── temporal/
│   ├── evaluation/
│   └── explainability/
│
├── experiments/
│   ├── configs/
│   ├── results/
│   └── notebooks/
│
├── reports/
├── docs/
├── tests/
├── README.md
└── docker-compose.yml

Use:
- Git.
- Reproducible random seeds.
- Configuration files.
- Modular code.
- Unit tests.
- Experiment tracking.
- Docker.
- GitHub Actions where appropriate.

# 12. IMPORTANT RESEARCH CONSTRAINTS

These requirements are non-negotiable.

1. No hardcoded predictions.
2. No mock ML results.
3. No fabricated benchmark scores.
4. No data leakage.
5. No treating unknown labels as legitimate.
6. No claiming GNNs outperform traditional ML without experimental evidence.
7. No unnecessary API-based LLM wrappers.
8. No fake real-time blockchain monitoring.
9. No misleading claims that this is a production AML compliance system.

This is an offline research and demonstration platform using historical transaction data, not a real-world financial compliance product.

# 13. DEPLOYMENT

The project should be deployable.

Suggested:
- Frontend: Vercel.
- Backend: Render or another suitable Python hosting service.
- ML experimentation: Google Colab or local GPU.
- Model artifacts: Separate downloadable or hosted checkpoints.

Optimize inference and avoid loading huge graph structures into memory unnecessarily.

# 14. DEVELOPMENT ROADMAP

Build this in phases.

Phase 1: Environment setup and dataset exploration.

Phase 2: Data preprocessing and leakage-safe splits.

Phase 3: Traditional ML baselines.

Phase 4: GCN, GraphSAGE and GAT.

Phase 5: Temporal graph learning.

Phase 6: Evaluation and ablation studies.

Phase 7: Explainability.

Phase 8: FastAPI backend.

Phase 9: Next.js frontend.

Phase 10: Deployment, testing and documentation.

Do not skip the ML experiments to rush into frontend development.

# 15. WHAT I EXPECT FROM YOU

Before writing code:

1. Analyze the complete project requirements.
2. Identify technical risks.
3. Propose the most suitable architecture.
4. Explain the dataset preprocessing strategy.
5. Explain how to prevent temporal and graph leakage.
6. Propose the evaluation methodology.
7. Identify which components can run on free Google Colab GPUs.
8. Estimate computational requirements.
9. Provide a phased implementation plan.

Then start implementation one phase at a time.

At each phase:
- Explain what we are building.
- Create the necessary files.
- Write complete working code.
- Explain how to execute it.
- Run tests.
- Verify outputs.
- Document progress.

If an architecture is unnecessarily complicated, suggest a simpler scientifically valid alternative.

**FINAL OBJECTIVE:**

Create GraphGuard as a technically credible Graph Machine Learning research project with a polished interactive application, reproducible benchmarks, genuine experiments, explainable predictions, and a professional GitHub repository.

The project should demonstrate that I understand ML beyond simply importing models and calling `.fit()`.

Prioritize scientific correctness, engineering quality, and measurable results over flashy features.