| Ablation Dimension   | Variant / Condition                            | PR-AUC (mean ± std)   | Δ PR-AUC vs Ref [95% CI]   | F1 (mean ± std)   |
|:---------------------|:-----------------------------------------------|:----------------------|:---------------------------|:------------------|
| 1. Features          | XGB (all features, 165d) [Reference]           | 0.8001 ± 0.0008       | — [Reference]              | 0.7991 ± 0.0025   |
| 1. Features          | XGB (aggregated only, 72d)                     | 0.6364 ± 0.0207       | -0.1494 [-0.2575, -0.0836] | 0.5716 ± 0.0309   |
| 1. Features          | GCN (all 2-layer undirected, 165d) [Reference] | 0.6228 ± 0.0225       | — [Reference]              | 0.5941 ± 0.0504   |
| 1. Features          | GCN (aggregated only, 72d)                     | 0.4151 ± 0.0469       | -0.2080 [-0.2988, -0.1027] | 0.4402 ± 0.0251   |
| 2. Topology Only     | GCN (all 2-layer undirected, 165d) [Reference] | 0.6228 ± 0.0225       | — [Reference]              | 0.5941 ± 0.0504   |
| 2. Topology Only     | GCN (constant 1s, no features)                 | 0.0765 ± 0.0009       | -0.5425 [-0.6901, -0.2975] | 0.1031 ± 0.0544   |
| 3. Neighbour Hops    | GCN 2-hop (2 layers, 165d) [Reference]         | 0.6228 ± 0.0225       | — [Reference]              | 0.5941 ± 0.0504   |
| 3. Neighbour Hops    | GCN 1-hop (1 layer)                            | 0.2029 ± 0.0123       | -0.4126 [-0.5259, -0.2707] | 0.3135 ± 0.0172   |
| 4. Edge Direction    | GCN undirected (symmetrised) [Reference]       | 0.6228 ± 0.0225       | — [Reference]              | 0.5941 ± 0.0504   |
| 4. Edge Direction    | GCN (directed edges)                           | 0.4671 ± 0.0212       | -0.1303 [-0.1764, -0.0457] | 0.4662 ± 0.0242   |
| 5. Temporal Dynamics | EvolveGCN (evolving weights, 165d) [Reference] | 0.2211 ± 0.0474       | — [Reference]              | 0.1871 ± 0.1627   |
| 5. Temporal Dynamics | EvolveGCN (frozen weights)                     | 0.6348 ± 0.0110       | +0.2045 [+0.1253, +0.2878] | 0.5951 ± 0.0142   |