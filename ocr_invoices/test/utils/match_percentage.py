from .levenshtein_distance import levenshtein_distance


def match_percentage(s1, s2):
    if not s1 or not s2:
        return 0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:  # Avoid division by zero for two empty strings
        return 100
    similarity_percentage = (1 - distance / max_len) * 100
    return similarity_percentage


def mean_match_percentage(predicted_items, ocr_items):
    all_match_percentages = [
        match_percentage(pred_item, ocr_item)
        for pred_item in predicted_items
        for ocr_item in ocr_items
    ]
    # from all_match_percentages, keep only the N with highest values where N is the lenght of predicted_items
    all_match_percentages = sorted(all_match_percentages, reverse=True)[
        : len(predicted_items)
    ]
    if not all_match_percentages:  # Avoid division by zero
        return 0
    return sum(all_match_percentages) / len(all_match_percentages)
