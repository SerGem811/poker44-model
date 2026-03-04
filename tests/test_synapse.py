from __future__ import annotations

from poker44.validator.synapse import DetectionSynapse


def test_detection_synapse_keeps_chunks_in_body_fields():
    synapse = DetectionSynapse(chunks=[[{"label": "human"}], [{"label": "bot"}]])

    assert synapse.required_hash_fields == ["chunks"]
    assert len(synapse.chunks) == 2
    assert synapse.risk_scores is None
    assert synapse.predictions is None


def test_detection_synapse_deserialize_is_identity():
    synapse = DetectionSynapse(chunks=[[{"label": "human"}]])

    assert synapse.deserialize() is synapse
