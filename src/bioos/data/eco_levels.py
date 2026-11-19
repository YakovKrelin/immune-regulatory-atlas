ECO_LEVELS: dict[str, dict[str, object]] = {
    "A": {"code": "A", "label": "Direct experimental confirmation in humans", "weight": 1.0},
    "B": {"code": "B", "label": "Experimental confirmation in model system", "weight": 0.85},
    "C": {"code": "C", "label": "Consistent observational evidence", "weight": 0.70},
    "D": {"code": "D", "label": "Computational prediction", "weight": 0.45},
    "E": {"code": "E", "label": "Plausible hypothesis", "weight": 0.35},
    "F": {"code": "F", "label": "Conflicting evidence", "weight": 0.40},
    "G": {"code": "G", "label": "Retracted knowledge", "weight": 0.0},
}

ORGANISM_SYSTEMS: list[dict[str, object]] = [
    {
        "id": "immunity",
        "title": "Immunity",
        "domains": [
            "Innate",
            "Adaptive",
            "Interferon Response",
            "Inflammation",
            "Tolerance",
            "Resolution of Inflammation",
        ],
    },
    {
        "id": "metabolism",
        "title": "Metabolism",
        "domains": [
            "Carbohydrate",
            "Lipid",
            "Protein",
            "Energetics",
            "mTOR",
            "AMPK",
            "Mitochondria",
        ],
    },
    {
        "id": "endocrinology",
        "title": "Endocrinology",
        "domains": [
            "Cortisol",
            "Insulin",
            "Leptin",
            "Thyroid Hormones",
            "Sex Hormones",
            "Vitamin D",
        ],
    },
    {
        "id": "neuro",
        "title": "Nervous System",
        "domains": [
            "Sympathetic",
            "Parasympathetic",
            "Vagus Nerve",
            "Microglia",
            "Stress",
            "Neurotransmitters",
        ],
    },
    {
        "id": "hypoxia",
        "title": "Hypoxia",
        "domains": [
            "HIF-1alpha / HIF-2alpha",
            "Metabolic Shift",
            "Erythropoiesis",
            "Vascular Remodeling",
        ],
    },
]

CONTEXT_FILTERS: dict[str, list[str]] = {
    "pathologies": [
        "All",
        "Viral Infections",
        "Bacterial Infections",
        "Oncology",
        "Autoimmune Conditions",
        "Inflammatory Bowel Disease",
    ],
    "tissues": ["All", "Intestine", "Liver", "Blood"],
    "physiologies": ["All", "Normoxia", "Hypoxia", "Stress"],
}