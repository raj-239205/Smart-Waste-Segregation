"""Waste categories, segregation guidance, and display colors."""

WASTE_INFO = {
    "plastic": {
        "category": "Recyclable",
        "bin_color": "Blue",
        "disposal": "Empty and rinse containers when practical, then place them in the recyclable/plastic waste stream.",
        "examples": ["Water bottles", "Food containers", "Packaging"],
    },
    "paper": {
        "category": "Recyclable",
        "bin_color": "Blue",
        "disposal": "Keep paper dry and clean before placing it in the paper/recyclable waste stream.",
        "examples": ["Newspapers", "Cardboard", "Office paper"],
    },
    "metal": {
        "category": "Recyclable",
        "bin_color": "Blue",
        "disposal": "Empty and rinse metal containers where practical, then place them with recyclable metal waste.",
        "examples": ["Cans", "Tin containers", "Aluminium foil"],
    },
    "glass": {
        "category": "Recyclable",
        "bin_color": "Green",
        "disposal": "Handle broken glass carefully and follow your local glass-recycling guidance. Do not mix sharp glass with loose household waste.",
        "examples": ["Glass bottles", "Jars", "Glass containers"],
    },
    "organic": {
        "category": "Biodegradable",
        "bin_color": "Green",
        "disposal": "Place food and other compostable organic material in the appropriate wet/organic waste stream.",
        "examples": ["Food scraps", "Fruit and vegetable peels", "Garden waste"],
    },
    "unknown": {
        "category": "Unclassified",
        "bin_color": "Grey",
        "disposal": "The model is not sufficiently confident. Check the item manually instead of relying on an automatic segregation decision.",
        "examples": [],
    },
}

# RGB colors used consistently by the Streamlit and OpenCV interfaces.
CLASS_COLORS = {
    "plastic": (34, 197, 94),
    "paper": (59, 130, 246),
    "metal": (239, 68, 68),
    "glass": (168, 85, 247),
    "organic": (245, 158, 11),
    "unknown": (107, 114, 128),
}


def get_waste_info(class_name):
    """Return segregation information for a supported class."""
    key = str(class_name or "unknown").lower().strip()
    return WASTE_INFO.get(key, WASTE_INFO["unknown"])
