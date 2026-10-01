"""
tests — HITGNN test suite.

Test modules:
    - test_synthetic: Synthetic data generation validation
    - test_graph_builder: Graph construction correctness
    - test_missingness_encoder: Missingness encoder shape and behavior
    - test_te_gat: TE-GAT forward pass and time-in-attention
    - test_hierarchical_pool: Pooling output shapes
    - test_population_graph: Inductive protocol correctness
    - test_model_e2e: Full model forward/backward pass
    - test_no_leakage: Temporal and label leakage prevention
"""
