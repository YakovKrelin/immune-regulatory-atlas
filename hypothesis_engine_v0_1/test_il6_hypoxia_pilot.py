from __future__ import annotations

import json
from pathlib import Path

from hypothesis_engine_v0_1.il1_hypoxia_pilot import evaluate_il6_hypoxia_pilot


def test_bioos_test_001_sample_json_weak_inference() -> None:
    sample_path = Path(__file__).with_name("sample_il6_hypoxia_bioos_test_001.json")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    result = evaluate_il6_hypoxia_pilot(payload)

    assert result["mode"] == "weak_inference"
    assert result["hypotheses"]["H1"]["data_status"] == "compatible"
    assert result["hypotheses"]["H2"]["data_status"] == "not_supported_by_data"
    assert result["hypotheses"]["H3"]["data_status"] == "partial_unconfirmed"
    assert result["hypotheses"]["H4"]["data_status"] == "unconfirmed_cell_composition"
    assert result["hypotheses"]["H5"]["active"] is True

    assert result["il1_pathway"]["il1b_priming_status"] == "observed_increase"
    assert result["il1_pathway"]["il1b_processing_status"] == "unassessed_processing"
    assert result["il1_pathway"]["il1b_secretion_status"] == "not_assessed"
    assert result["il1_pathway"]["il1_alpha_status"] == "not_assessed"

    assert result["pairwise_relations"][0]["hypotheses"] == ["H1", "H3"]
    assert result["pairwise_relations"][0]["relation"] == "non_exclusive_not_separable"
    assert result["discriminating_measurements"][0]["priority"] == 1
    assert "Secreted IL6" in result["discriminating_measurements"][0]["measurement"]


def test_hard_abstention_when_critical_context_missing() -> None:
    payload = {
        "query_id": "BIOOS-ABSTAIN-001",
        "target": {"entity": "IL6", "level": "mRNA"},
        "context": {"organism": "Homo sapiens"},
        "observations": [{"entity": "IL6", "level": "mRNA", "direction": "increase"}],
    }
    result = evaluate_il6_hypoxia_pilot(payload)
    assert result["mode"] == "hard_abstention"
    assert "context.cell_type" in result["missing_requirements"]
    assert "context.condition" in result["missing_requirements"]


def test_conditional_inference_with_secreted_il6_and_mature_il1b() -> None:
    payload = {
        "query_id": "BIOOS-COND-001",
        "scenario": "IL6_hypoxia_macrophage",
        "data_type": "targeted_assays",
        "target": {"entity": "IL6", "level": "secreted_protein"},
        "context": {
            "organism": "Homo sapiens",
            "cell_type": "Macrophage",
            "condition": {
                "oxygen": "Hypoxia",
                "oxygen_level": "1% O2",
                "stimulus": "None",
                "priming": "None",
                "co_stimulus": "None",
            },
            "timepoint": "12h",
            "sample_type": "Culture supernatant",
        },
        "observations": [
            {
                "observation_id": "OBS-C01",
                "entity": "IL6",
                "assay": "ELISA",
                "level": "secreted_protein",
                "direction": "increase",
                "value": 145.0,
                "unit": "pg/mL",
            },
            {
                "observation_id": "OBS-C02",
                "entity": "HIF1A",
                "assay": "qPCR",
                "level": "mRNA",
                "direction": "increase",
            },
            {
                "observation_id": "OBS-C03",
                "entity": "IL1B",
                "assay": "qPCR",
                "level": "mRNA",
                "direction": "increase",
            },
            {
                "observation_id": "OBS-C04",
                "entity": "IL1B",
                "assay": "ELISA",
                "level": "secreted_protein",
                "direction": "increase",
            },
            {
                "observation_id": "OBS-C05",
                "entity": "IL1A",
                "assay": "ELISA",
                "level": "secreted_protein",
                "direction": "increase",
            },
        ],
    }

    result = evaluate_il6_hypoxia_pilot(payload)

    assert result["mode"] == "conditional_inference"
    assert result["hypotheses"]["H1"]["data_status"] == "compatible"
    assert result["hypotheses"]["H2"]["data_status"] == "not_supported_by_data"
    assert result["hypotheses"]["H3"]["data_status"] == "supported"
    assert result["hypotheses"]["H4"]["data_status"] == "not_assessed"
    assert result["hypotheses"]["H5"]["active"] is False

    assert result["il1_pathway"]["il1b_priming_status"] == "observed_increase"
    assert result["il1_pathway"]["il1b_processing_status"] == "observed"
    assert result["il1_pathway"]["il1b_secretion_status"] == "observed"
    assert result["il1_pathway"]["il1_alpha_status"] == "observed_increase"

def test_il1_pathway_processing_caspase1_and_tlr_stimulus() -> None:
    payload = {
        "query_id": "BIOOS-IL1-FULL-001",
        "scenario": "IL6_hypoxia_macrophage",
        "data_type": "targeted_assays",
        "target": {"entity": "IL6", "level": "secreted_protein"},
        "context": {
            "organism": "Homo sapiens",
            "cell_type": "Macrophage",
            "condition": {
                "oxygen": "Hypoxia",
                "stimulus": "LPS",
                "co_stimulus": "None",
            },
        },
        "observations": [
            {"entity": "IL6", "assay": "ELISA", "level": "secreted_protein", "direction": "increase"},
            {"entity": "HIF1A", "assay": "qPCR", "level": "mRNA", "direction": "increase"},
            {"entity": "IL1B", "assay": "Western blot", "level": "mature_protein", "direction": "increase"},
            {"entity": "IL1B", "assay": "ELISA", "level": "secreted_protein", "direction": "increase"},
            {"entity": "CASP1", "assay": "caspase-1 activity", "level": "activity", "direction": "increase"},
            {"entity": "IL1A", "assay": "ELISA", "level": "secreted_protein", "direction": "increase"},
        ],
    }
    result = evaluate_il6_hypoxia_pilot(payload)
    assert result["hypotheses"]["H2"]["data_status"] == "compatible"
    assert result["hypotheses"]["H3"]["data_status"] == "supported"
    assert result["il1_pathway"]["il1b_priming_status"] == "not_assessed"
    assert result["il1_pathway"]["il1b_secretion_status"] == "observed"


def test_hard_abstention_missing_target_organism_and_il6_obs() -> None:
    payload = {
        "query_id": "BIOOS-ABSTAIN-002",
        "context": {"cell_type": "Macrophage", "condition": {"oxygen": "Hypoxia"}},
        "observations": [],
    }
    result = evaluate_il6_hypoxia_pilot(payload)
    assert result["mode"] == "hard_abstention"
    assert "target.entity" in result["missing_requirements"]
    assert "context.organism" in result["missing_requirements"]
    assert "observations[IL6]" in result["missing_requirements"]

def test_il1_pathway_not_assessed_when_il1_entities_absent() -> None:
    payload = {
        "query_id": "BIOOS-IL1-NONE-001",
        "target": {"entity": "IL6", "level": "mRNA"},
        "context": {"organism": "Homo sapiens", "cell_type": "Macrophage", "condition": {"oxygen": "Hypoxia"}},
        "observations": [
            {"entity": "IL6", "level": "mRNA", "direction": "increase"},
            {"entity": "HIF1A", "level": "mRNA", "direction": "increase"},
        ],
    }
    result = evaluate_il6_hypoxia_pilot(payload)
    assert result["il1_pathway"]["il1b_priming_status"] == "not_assessed"
    assert result["il1_pathway"]["il11_alpha_status"] if False else result["il1_pathway"]["il1_alpha_status"] == "not_assessed"

def test_run_hypothesis_engine_populates_il1_pathway_and_pilot_hypotheses() -> None:
    from hypothesis_engine_v0_1.engine import run_hypothesis_engine
    from hypothesis_engine_v0_1.schemas import HypothesisEngineInput

    payload = HypothesisEngineInput.model_validate(
        {
            "query_id": "BIOOS-INT-001",
            "scenario": "IL6_hypoxia_macrophage",
            "data_type": "bulk_rna_seq",
            "target": {"entity": "IL6", "level": "mRNA"},
            "context": {
                "organism": "Homo sapiens",
                "cell_type": "Macrophage",
                "timepoint": "12h",
                "condition": {"oxygen": "Hypoxia", "oxygen_level": "1% O2", "stimulus": "None"},
            },
            "observations": [
                {"observation_id": "O1", "entity": "IL6", "assay": "RNA-seq", "level": "mRNA", "direction": "up", "value": 1.8, "unit": "log2FC", "fdr": 0.01},
                {"observation_id": "O2", "entity": "HIF1A", "assay": "RNA-seq", "level": "mRNA", "direction": "up"},
                {"observation_id": "O3", "entity": "IL1B", "assay": "RNA-seq", "level": "mRNA", "direction": "up"},
            ],
        }
    )
    out = run_hypothesis_engine(payload)
    assert out.il1_pathway is not None
    assert out.il1_pathway.il1b_priming_status == "observed"
    assert out.pilot_hypotheses is not None
    assert out.pilot_hypotheses["H1"].name == "HIF-associated IL6 regulation"
    assert out.pilot_hypotheses["H4"].name == "Cell-composition shift"
    assert out.pilot_hypotheses["H5"].active is True