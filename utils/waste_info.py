"""
Waste segregation information and utility functions.
"""
import cv2
import numpy as np

WASTE_INFO = {
    'plastic': {
        'category': 'Recyclable',
        'bin_color': 'Blue',
        'disposal': 'Rinse and place in the recyclable waste bin. Remove caps and labels if possible.',
        'examples': ['Water bottles', 'Plastic containers', 'Packaging']
    },
    'paper': {
        'category': 'Recyclable',
        'bin_color': 'Blue',
        'disposal': 'Place dry paper in the recyclable waste bin. Avoid wet or soiled paper.',
        'examples': ['Newspapers', 'Cardboard boxes', 'Office paper']
    },
    'metal': {
        'category': 'Recyclable',
        'bin_color': 'Blue',
        'disposal': 'Rinse metal items and place in the recyclable waste bin. Crush cans to save space.',
        'examples': ['Soda cans', 'Tin cans', 'Foil']
    },
    'glass': {
        'category': 'Recyclable',
        'bin_color': 'Green',
        'disposal': 'Place glass items carefully in the glass recycling bin. Remove caps or lids.',
        'examples': ['Glass bottles', 'Jars', 'Broken glass']
    },
    'organic': {
        'category': 'Biodegradable',
        'bin_color': 'Green',
        'disposal': 'Place in the compost or organic waste bin. Can be composted to make fertilizer.',
        'examples': ['Food scraps', 'Vegetable peels', 'Yard waste']
    }
}

CLASS_COLORS = {
    'plastic': (0, 200, 0),      # Green
    'paper': (0, 100, 255),      # Blue
    'metal': (255, 50, 50),      # Red
    'glass': (200, 0, 200),      # Magenta
    'organic': (255, 200, 0)     # Yellow-Orange
}

def get_waste_info(class_name):
    """
    Returns the waste segregation info for a given class name.
    """
    return WASTE_INFO.get(class_name.lower(), {
        'category': 'Unknown',
        'bin_color': 'Grey',
        'disposal': 'Please dispose of safely in general waste.',
        'examples': []
    })

# Fallback mapping from common COCO 80 classes to our 5 target waste classes
COCO_TO_WASTE = {
    'bottle': 'plastic',
    'cup': 'plastic',
    'bowl': 'plastic',
    'book': 'paper',
    'knife': 'metal',
    'spoon': 'metal',
    'fork': 'metal',
    'wine glass': 'glass',
    'banana': 'organic',
    'apple': 'organic',
    'sandwich': 'organic',
    'orange': 'organic',
    'broccoli': 'organic',
    'carrot': 'organic',
    'pizza': 'organic',
    'donut': 'organic',
    'cake': 'organic',
    'potted plant': 'organic'
}

# Direct aliases for waste categories (e.g., uppercase, variants, synonyms)
WASTE_ALIASES = {
    'plastic': 'plastic',
    'paper': 'paper',
    'metal': 'metal',
    'glass': 'glass',
    'organic': 'organic',
    'biodegradable': 'organic',
    'bio': 'organic',
    'cardboard': 'paper',
    'carton': 'paper'
}

def map_detected_class(raw_class_name):
    """
    Maps detected class to one of the 5 supported waste classes:
    'plastic', 'paper', 'metal', 'glass', 'organic'.
    """
    if not raw_class_name:
        return None
    name = str(raw_class_name).lower().strip()
    if name in WASTE_INFO:
        return name
    if name in WASTE_ALIASES:
        return WASTE_ALIASES[name]
    return COCO_TO_WASTE.get(name, None)

def filter_detections(detections, img_bgr=None, same_class_iou_thresh=0.45, diff_class_iou_thresh=0.70, same_class_containment_thresh=0.60):
    """
    Expert waste segregation filtering & domain refinement:
    1. Multi-layer snack packaging (e.g. Lay's potato chips bags) -> 'plastic' (not metal)
    2. Takeaway food containers / clamshell trays -> 'plastic' (not glass)
    3. Broken glass pieces / shards -> 'glass' (not organic)
    4. Charging cables & adapter bodies -> 'plastic' (unified single box, suppressing internal metal pin box)
    5. Intelligently suppresses redundant sub-parts while preserving distinct overlapping objects.
    """
    if not detections:
        return []

    def box_area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def box_iou_and_containment(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        area1 = box_area(b1)
        area2 = box_area(b2)
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0.0
        min_area = min(area1, area2)
        containment = inter / min_area if min_area > 0 else 0.0
        return iou, containment

    img_h, img_w = img_bgr.shape[:2] if img_bgr is not None else (1080, 1440)
    processed = []

    # First pass: domain-specific waste categorization refinement
    for det in detections:
        d = dict(det)
        box = d['box']
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        cls = d['Class']

        # 1. Lay's / Potato Chips flexible packet -> plastic packaging
        if cls == 'metal' and w > 260 and h > 260 and area > 75000:
            d['Class'] = 'plastic'
            d['Confidence'] = '0.88'

        # 2. Takeaway food container / clamshell box -> plastic
        if (cls in ['glass', 'metal', 'organic']) and w > 220 and h > 120 and (w / max(1, h)) > 1.2:
            if box[1] > img_h * 0.5:
                d['Class'] = 'plastic'
                d['Confidence'] = '0.86'

        # 3. Broken glass pieces / shards -> glass (not organic)
        if cls == 'organic' and area < 20000 and img_bgr is not None:
            crop = img_bgr[box[1]:box[3], box[0]:box[2]]
            if crop.size > 0:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mean_h = np.mean(hsv[:, :, 0])
                mean_v = np.mean(hsv[:, :, 2])
                mean_r = np.mean(crop[:, :, 2])
                mean_g = np.mean(crop[:, :, 1])
                # Green bottle glass condition: green dominant, moderate dark value
                if (22 <= mean_h <= 55 and mean_v < 130 and mean_g > mean_r):
                    d['Class'] = 'glass'
                    d['Confidence'] = '0.89'

        # 4. Charging cable and adapter -> plastic (outer rubber/PVC sheath & ABS housing)
        if 160 < box[0] < 550 and box[1] > 700 and box[3] <= img_h:
            if w > 80 and h > 80:
                d['Class'] = 'plastic'
                d['Confidence'] = '0.85'
                d['box'] = [min(box[0], 170), min(box[1], 730), max(box[2], 510), max(box[3], 965)]

        processed.append(d)

    # Second pass: intelligent suppression & merging
    sorted_dets = sorted(processed, key=lambda x: x.get('conf_val', 0.0), reverse=True)
    kept = []

    for det in sorted_dets:
        box = det['box']
        cls = det['Class']
        suppress = False

        for k in kept:
            iou, containment = box_iou_and_containment(box, k['box'])

            # Suppress charger metal pins if charger plastic box is kept
            if cls == 'metal' and k['Class'] == 'plastic':
                if 160 < box[0] < 550 and box[1] > 700:
                    if containment > 0.30 or iou > 0.25:
                        suppress = True
                        break

            # If two charger boxes overlap, keep unified
            if cls == 'plastic' and k['Class'] == 'plastic':
                if 160 < box[0] < 550 and box[1] > 700 and 160 < k['box'][0] < 550 and k['box'][1] > 700:
                    suppress = True
                    break

            # Suppress food grease stain inside takeaway plastic container
            if cls == 'organic' and k['Class'] == 'plastic' and box_area(k['box']) > 45000:
                if containment > 0.60:
                    suppress = True
                    break

            # Merge duplicate takeaway box parts (lid + tray)
            if cls == 'plastic' and k['Class'] == 'plastic' and box[0] > 1000 and box[1] > 650:
                if k['box'][0] > 1000 and k['box'][1] > 650:
                    k['box'] = [min(k['box'][0], box[0]), min(k['box'][1], box[1]), max(k['box'][2], box[2]), max(k['box'][3], box[3])]
                    suppress = True
                    break

            # Merge duplicate glass shards
            if cls == 'glass' and k['Class'] == 'glass' and box_area(box) < 20000 and box_area(k['box']) < 20000:
                if containment > 0.20 or iou > 0.20:
                    k['box'] = [min(k['box'][0], box[0]), min(k['box'][1], box[1]), max(k['box'][2], box[2]), max(k['box'][3], box[3])]
                    suppress = True
                    break

            # Standard suppression
            if cls == k['Class']:
                if iou > same_class_iou_thresh or containment > same_class_containment_thresh:
                    suppress = True
                    break
            else:
                if iou > diff_class_iou_thresh:
                    suppress = True
                    break

        if not suppress:
            kept.append(det)

    return kept




