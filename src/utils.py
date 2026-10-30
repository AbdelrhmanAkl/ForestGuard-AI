COVER_TYPES = {
    1: {
        "name": "Spruce/Fir",
        "description": "Spruce/Fir forest cover."
    },
    2: {
        "name": "Lodgepole Pine",
        "description": "Lodgepole Pine forest cover."
    },
    3: {
        "name": "Ponderosa Pine",
        "description": "Ponderosa Pine forest cover."
    },
    4: {
        "name": "Cottonwood/Willow",
        "description": "Cottonwood/Willow forest cover."
    },
    5: {
        "name": "Aspen",
        "description": "Aspen forest cover."
    },
    6: {
        "name": "Douglas-fir",
        "description": "Douglas-fir forest cover."
    },
    7: {
        "name": "Krummholz",
        "description": "Krummholz forest cover."
    },
}


def get_cover_type(class_id):
    """Return descriptive information for a predicted cover type."""
    return COVER_TYPES.get(
        int(class_id),
        {"name": "Unknown", "description": "Unknown cover type."}
    )
