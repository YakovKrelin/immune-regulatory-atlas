from bioos.engine.filters import filter_assertions


def test_filter_assertions_eco_and_evidence_fallback():
    items = [
        "not-a-dict",
        {"id": "1", "eco": "A", "context": {"organism": "Homo sapiens", "tissue": "Blood"}},
        {"id": "2", "evidence": {"level": "C", "cell_type": "Macrophage"}},
        {"id": "3", "eco_level": "F", "evidence": {"modality": "Proteomics"}},
        {"id": "4", "evidence_level": "G", "measurement_level": "RNA-seq"},
        {"id": "5", "context": {}},
    ]

    assert [x["id"] for x in filter_assertions(items, eco_levels=["a,c"])] == ["1", "2"]
    assert [x["id"] for x in filter_assertions(items, min_eco="C")] == ["1", "2"]
    assert [x["id"] for x in filter_assertions(items, min_eco="UNKNOWN")] == ["1", "2", "3", "4", "5"]
    assert [x["id"] for x in filter_assertions(items, modality="proteomics")] == ["3"]
    assert [x["id"] for x in filter_assertions(items, modality="rna-seq")] == ["4"]


def test_filter_assertions_all_context_fields_and_conflicts():
    items = [
        {
            "id": "c1",
            "eco": "B",
            "conflict_status": "direct_contradiction",
            "context": {
                "organism": "Homo sapiens",
                "cell_type": "T cell",
                "tissue": "Spleen",
                "pathology": "Sepsis",
                "physiology": "Acute response",
                "stimulus": "LPS",
                "modality": "RNA-seq",
                "phase": "Early",
                "timepoint": "4h",
                "translational_status": "Preclinical",
            },
        },
        {
            "id": "c2",
            "eco": "D",
            "contradicts": ["c1"],
            "evidence": {
                "organism": "Mus musculus",
                "cell_type": "Monocyte",
                "tissue": "Lung",
                "pathology": "Fibrosis",
                "physiology": "Repair",
                "stimulus": "Hypoxia",
                "phase": "Late",
                "timepoint": "24h",
                "translational_status": "Clinical",
            },
        },
        {
            "id": "no_conflict",
            "eco": "A",
            "has_conflict": False,
            "organism": "Homo sapiens",
        },
    ]

    assert [x["id"] for x in filter_assertions(items, conflict_only=True)] == ["c1", "c2"]
    assert [
        x["id"]
        for x in filter_assertions(
            items,
            organism="homo sapiens",
            cell_type="t cell",
            tissue="spleen",
            pathology="sepsis",
            physiology="acute response",
            stimulus="lps",
            modality="rna-seq",
            phase="early",
            timepoint="4h",
            translational_status="preclinical",
            conflict_only=True,
        )
    ] == ["c1"]

    assert [
        x["id"]
        for x in filter_assertions(
            items,
            organism="mus musculus",
            cell_type="monocyte",
            tissue="lung",
            pathology="fibrosis",
            physiology="repair",
            stimulus="hypoxia",
            phase="late",
            timepoint="24h",
            translational_status="clinical",
        )
    ] == ["c2"]

    assert [x["id"] for x in filter_assertions(items, organism="homo sapiens", conflict_only=False)] == [
        "c1",
        "no_conflict",
    ]


def test_filter_assertions_remaining_branches():
    items = [
        {"id": "f1", "evidence": {"eco": "E", "pathology": "Sepsis", "physiology": "Inflammation", "stimulus": "LPS", "phase": "Acute", "timepoint": "2h", "translational_status": "InVivo"}},
        {"id": "f2", "eco": "B", "context": {"pathology": "Fibrosis"}, "evidence": {"physiology": "Repair", "stimulus": "Hypoxia", "phase": "Chronic", "timepoint": "48h", "translational_status": "Clinical"}},
        {"id": "f3", "eco": "C", "context": {"stimulus": "IFN"}},
    ]

    assert [x["id"] for x in filter_assertions(items, eco_levels=["", "e"])] == ["f1"]
    assert [
        x["id"]
        for x in filter_assertions(
            items,
            pathology="sepsis",
            physiology="inflammation",
            stimulus="lps",
            phase="acute",
            timepoint="2h",
            translational_status="invivo",
            modality="nonexistent",
        )
    ] == []
    assert [
        x["id"]
        for x in filter_assertions(
            items,
            pathology="sepsis",
            physiology="inflammation",
            stimulus="lps",
            phase="acute",
            timepoint="2h",
            translational_status="invivo",
        )
    ] == ["f1"]
    assert [x["id"] for x in filter_assertions(items, stimulus="ifn")] == ["f3"]
    assert [x["id"] for x in filter_assertions(items, conflict_only=True)] == []
