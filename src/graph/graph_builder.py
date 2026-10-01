"""
src.graph.graph_builder — Per-Patient Temporal Event Graph Construction.

Converts a list of :class:`ClinicalEvent` objects into a homogeneous
``torch_geometric.data.Data`` graph with:

    - Explicit tensor features (node_type, concept_id, time, numeric_value)
    - ``edge_type`` attributes (integer indices: 0=TEMPORAL_ADJACENCY,
      1=SAME_CONCEPT_RECURRENCE, 2=SEMANTIC, 3=CROSS_MODALITY)
    - ``edge_delta_t`` attributes (relative time gaps in hours)

The four edge types encode:
    A. TEMPORAL_ADJACENCY: consecutive events within the same category
    B. SAME_CONCEPT_RECURRENCE: consecutive observations of the same concept
    C. SEMANTIC: clinically related concepts (temporally nearest)
    D. CROSS_MODALITY: structured event ↔ clinical concept within ±30 min
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from src.data.synthetic_generator import ClinicalEvent, PatientRecord
from src.data.terminology import ClinicalTerminology
from src.data.vocab import ConceptVocabulary
from src.preprocessing.event_normalizer import EventNormalizer


# Edge type constants
EDGE_TYPE_TEMPORAL_ADJACENCY = 0
EDGE_TYPE_SAME_CONCEPT = 1
EDGE_TYPE_SEMANTIC = 2
EDGE_TYPE_CROSS_MODALITY = 3

NUM_EDGE_TYPES = 4

# Node type constants
NODE_TYPE_MAP = {
    "LAB": 0,
    "VITAL": 1,
    "MEDICATION": 2,
    "PROCEDURE": 3,
    "DIAGNOSIS": 4,
    "CLINICAL_CONCEPT": 5,
}

def get_node_type_id(event_type: str) -> int:
    if event_type not in NODE_TYPE_MAP:
        raise ValueError(f"Unknown event type: {event_type}")
    return NODE_TYPE_MAP[event_type]


class PatientGraphBuilder:
    """Build a temporal event graph for a single patient.

    Parameters
    ----------
    vocab : ConceptVocabulary
        Canonical concept vocabulary for embedding lookups.
    terminology : ClinicalTerminology
        Semantic relationship dictionary for SEMANTIC edges.
    normalizer : EventNormalizer
        Trained normalizer for numeric event values.
    config : dict
        The ``graph`` section of the configuration.
    """

    def __init__(
        self,
        vocab: ConceptVocabulary,
        terminology: ClinicalTerminology,
        normalizer: EventNormalizer,
        config: dict,
    ) -> None:
        self.vocab = vocab
        self.terminology = terminology
        self.normalizer = normalizer
        self.config = config
        self.cross_modality_window_hours = config.get("cross_modality_window_min", 30.0) / 60.0
        
        ablation_cfg = config.get("ablation", {})
        self.no_temporal_edges = ablation_cfg.get("no_temporal_edges", False)
        self.no_semantic_edges = ablation_cfg.get("no_semantic_edges", False)

    def build(self, patient: PatientRecord) -> Data:
        """Convert a patient record into a PyG Data object."""
        import warnings
        # 1. Filter out events strictly after t0
        future_events = [e for e in patient.events if e.timestamp > 0.0]
        if future_events:
            warnings.warn(
                f"Filtered {len(future_events)} future events (timestamp > t0) for patient {patient.patient_id}",
                UserWarning
            )
        events = [e for e in patient.events if e.timestamp <= 0.0]
        
        # 2. Deterministically sort events: timestamp -> node_type -> concept_id -> original index
        # We attach an original index to ensure stable sorting for duplicate timestamps
        annotated_events = [
            (e.timestamp, get_node_type_id(e.event_type), e.concept_id, i, e)
            for i, e in enumerate(events)
        ]
        annotated_events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        events = [x[4] for x in annotated_events]

        if not events:
            # Handle empty graph edge cases
            return Data(
                node_type=torch.empty((0,), dtype=torch.long),
                concept_id=torch.empty((0,), dtype=torch.long),
                time=torch.empty((0,), dtype=torch.float),
                relative_time=torch.empty((0,), dtype=torch.float),
                numeric_value=torch.empty((0,), dtype=torch.float),
                has_value=torch.empty((0,), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_type=torch.empty((0,), dtype=torch.long),
                edge_delta_t=torch.empty((0,), dtype=torch.float),
            )

        # 3. Create explicit node features
        node_types = []
        concept_ids = []
        times = []
        relative_times = []
        numeric_values = []
        has_values = []

        for e in events:
            node_types.append(get_node_type_id(e.event_type))
            concept_ids.append(e.concept_id)
            times.append(e.timestamp)
            # Assuming relative_time is just timestamp since t0=0
            relative_times.append(e.timestamp)
            
            if e.value is not None and isinstance(e.value, (int, float)):
                has_values.append(1.0)
                numeric_values.append(self.normalizer.transform(e.concept_name, e.value))
            else:
                has_values.append(0.0)
                numeric_values.append(0.0)

        # 4. Build edges
        edges_temp = self._to_dict([], [], [], EDGE_TYPE_TEMPORAL_ADJACENCY) if self.no_temporal_edges else self._build_temporal_adjacency_edges(events)
        edges_rec = self._to_dict([], [], [], EDGE_TYPE_SAME_CONCEPT) if self.no_temporal_edges else self._build_same_concept_edges(events)
        edges_sem = self._to_dict([], [], [], EDGE_TYPE_SEMANTIC) if self.no_semantic_edges else self._build_semantic_edges(events)
        edges_cross = self._build_cross_modality_edges(events)

        # Concatenate edge lists
        all_edge_index = []
        all_edge_type = []
        all_edge_dt = []

        for d in [edges_temp, edges_rec, edges_sem, edges_cross]:
            if d["edge_index"].size(1) > 0:
                all_edge_index.append(d["edge_index"])
                all_edge_type.append(d["edge_type"])
                all_edge_dt.append(d["edge_delta_t"])

        if all_edge_index:
            edge_index = torch.cat(all_edge_index, dim=1)
            edge_type = torch.cat(all_edge_type, dim=0)
            edge_delta_t = torch.cat(all_edge_dt, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            edge_delta_t = torch.empty((0,), dtype=torch.float)

        # 5. Post-processing: deduplicate exact (src, dst, type) and remove self-loops
        if edge_index.size(1) > 0:
            # Self loops removal
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]
            edge_delta_t = edge_delta_t[mask]

            # Deduplication
            if edge_index.size(1) > 0:
                max_node = len(events)
                max_type = NUM_EDGE_TYPES
                hash_val = edge_index[0] * (max_node * max_type) + edge_index[1] * max_type + edge_type
                # torch.unique does not return original indices, use numpy
                import numpy as np
                _, unique_idx = np.unique(hash_val.numpy(), return_index=True)
                edge_index = edge_index[:, unique_idx]
                edge_type = edge_type[unique_idx]
                edge_delta_t = edge_delta_t[unique_idx]

        # 6. Construct Data object
        data = Data(
            node_type=torch.tensor(node_types, dtype=torch.long),
            concept_id=torch.tensor(concept_ids, dtype=torch.long),
            time=torch.tensor(times, dtype=torch.float),
            relative_time=torch.tensor(relative_times, dtype=torch.float),
            numeric_value=torch.tensor(numeric_values, dtype=torch.float),
            has_value=torch.tensor(has_values, dtype=torch.float),
            edge_index=edge_index,
            edge_type=edge_type,
            edge_delta_t=edge_delta_t,
        )

        return data

    def _build_temporal_adjacency_edges(
        self, events: List[ClinicalEvent]
    ) -> Dict[str, torch.Tensor]:
        """Connect consecutive events within each event category."""
        src, dst, dt = [], [], []
        cat_indices: Dict[str, List[int]] = {}
        for i, e in enumerate(events):
            cat_indices.setdefault(e.event_type, []).append(i)

        for cat, indices in cat_indices.items():
            for k in range(len(indices) - 1):
                u = indices[k]
                v = indices[k + 1]
                src.append(u)
                dst.append(v)
                dt.append(events[v].timestamp - events[u].timestamp)

        return self._to_dict(src, dst, dt, EDGE_TYPE_TEMPORAL_ADJACENCY)

    def _build_same_concept_edges(
        self, events: List[ClinicalEvent]
    ) -> Dict[str, torch.Tensor]:
        """Connect consecutive observations of the same clinical concept."""
        src, dst, dt = [], [], []
        concept_indices: Dict[int, List[int]] = {}
        for i, e in enumerate(events):
            concept_indices.setdefault(e.concept_id, []).append(i)

        for cid, indices in concept_indices.items():
            for k in range(len(indices) - 1):
                u = indices[k]
                v = indices[k + 1]
                src.append(u)
                dst.append(v)
                dt.append(events[v].timestamp - events[u].timestamp)

        return self._to_dict(src, dst, dt, EDGE_TYPE_SAME_CONCEPT)

    def _build_semantic_edges(
        self, events: List[ClinicalEvent]
    ) -> Dict[str, torch.Tensor]:
        """Connect semantically related concepts using temporally nearest target."""
        src, dst, dt = [], [], []
        
        concept_name_to_indices: Dict[str, List[int]] = {}
        for i, e in enumerate(events):
            concept_name_to_indices.setdefault(e.concept_name, []).append(i)

        for u, e_u in enumerate(events):
            relations = self.terminology.get_relations_for(e_u.concept_name)
            if not relations:
                continue
                
            for rel in relations:
                target_name = rel.target_concept
                if target_name not in concept_name_to_indices:
                    continue
                
                target_indices = concept_name_to_indices[target_name]
                if not target_indices:
                    continue

                t_u = e_u.timestamp
                best_v = -1
                best_dt = float('inf')
                for v in target_indices:
                    if u == v:
                        continue
                    current_dt = abs(events[v].timestamp - t_u)
                    if current_dt < best_dt:
                        best_dt = current_dt
                        best_v = v
                    elif current_dt == best_dt and best_v == -1:
                        best_v = v
                
                if best_v != -1:
                    src.append(u)
                    dst.append(best_v)
                    dt.append(best_dt)

        return self._to_dict(src, dst, dt, EDGE_TYPE_SEMANTIC)

    def _build_cross_modality_edges(
        self, events: List[ClinicalEvent]
    ) -> Dict[str, torch.Tensor]:
        """Connect structured events to clinical concepts within window (nearest match)."""
        src, dst, dt = [], [], []

        concept_indices = [i for i, e in enumerate(events) if e.event_type == "CLINICAL_CONCEPT"]
        if not concept_indices:
            return self._to_dict([], [], [], EDGE_TYPE_CROSS_MODALITY)

        for u, e_u in enumerate(events):
            if e_u.event_type == "CLINICAL_CONCEPT":
                continue

            t_u = e_u.timestamp
            best_v = -1
            best_dt = float('inf')

            for v in concept_indices:
                current_dt = abs(events[v].timestamp - t_u)
                if current_dt <= self.cross_modality_window_hours:
                    if current_dt < best_dt:
                        best_dt = current_dt
                        best_v = v
                    elif current_dt == best_dt and best_v == -1:
                        best_v = v

            if best_v != -1:
                src.append(u)
                dst.append(best_v)
                dt.append(best_dt)

        return self._to_dict(src, dst, dt, EDGE_TYPE_CROSS_MODALITY)

    def _to_dict(self, src: List[int], dst: List[int], dt: List[float], edge_type: int) -> Dict[str, torch.Tensor]:
        if not src:
            return {
                "edge_index": torch.empty((2, 0), dtype=torch.long),
                "edge_type": torch.empty((0,), dtype=torch.long),
                "edge_delta_t": torch.empty((0,), dtype=torch.float),
            }
        return {
            "edge_index": torch.tensor([src, dst], dtype=torch.long),
            "edge_type": torch.full((len(src),), edge_type, dtype=torch.long),
            "edge_delta_t": torch.tensor(dt, dtype=torch.float),
        }


def validate_patient_graph(data: Data) -> None:
    """Validate graph strictly according to Phase 4 rules."""
    assert hasattr(data, 'edge_index'), "Missing edge_index"
    assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2, "edge_index must be [2, E]"
    
    num_nodes = data.node_type.size(0)
    num_edges = data.edge_index.size(1)

    if num_nodes == 0:
        return

    if num_edges > 0:
        assert data.edge_index.max() < num_nodes, "Invalid max edge index"
        assert data.edge_index.min() >= 0, "Invalid min edge index"

    assert (data.time <= 0.0).all(), "Found timestamp > t0"
    assert (data.node_type >= 0).all() and (data.node_type <= 5).all(), "Invalid node_type"
    assert data.node_type.dtype == torch.long, "node_type must be long"
    assert data.concept_id.dtype == torch.long, "concept_id must be long"

    if num_edges > 0:
        assert (data.edge_type >= 0).all() and (data.edge_type <= 3).all(), "Invalid edge_type"
        assert (data.edge_delta_t >= -1e-6).all(), "edge_delta_t must be >= 0"
        assert (data.edge_index[0] != data.edge_index[1]).all(), "Self loops found"
        
        hash_val = data.edge_index[0] * (num_nodes * NUM_EDGE_TYPES) + data.edge_index[1] * NUM_EDGE_TYPES + data.edge_type
        assert len(torch.unique(hash_val)) == num_edges, "Duplicate typed edges found"

    assert not hasattr(data, 'severity'), "latent severity found"
    
    if num_edges > 0:
        rec_mask = data.edge_type == EDGE_TYPE_SAME_CONCEPT
        if rec_mask.any():
            rec_edges = data.edge_index[:, rec_mask]
            src_concepts = data.concept_id[rec_edges[0]]
            dst_concepts = data.concept_id[rec_edges[1]]
            assert (src_concepts == dst_concepts).all(), "Recurrence edge connects different concepts"

    if num_edges > 0:
        temp_mask = data.edge_type == EDGE_TYPE_TEMPORAL_ADJACENCY
        if temp_mask.any():
            temp_edges = data.edge_index[:, temp_mask]
            src_times = data.time[temp_edges[0]]
            dst_times = data.time[temp_edges[1]]
            assert (dst_times >= src_times - 1e-6).all(), "Temporal adjacency edge points backward in time"
