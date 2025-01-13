import Levenshtein
import re

def align_blocks(block1, block2, window=None, threshold=3):
    """
    Aligns two blocks of text line-by-line based on the minimum Levenshtein distance
    within a fixed window to preserve line order. Missing lines are filled with empty strings.

    Args:
        block1 (list of str): The first block of text as a list of lines.
        block2 (list of str): The second block of text as a list of lines.
        window (int): The window size for comparing lines to keep alignment ordered.
        threshold (int): The max Levenshtein distance to consider lines similar.

    Returns:
        tuple: Aligned versions of block1 and block2 with the same number of lines.


    Note:
        - For alignment with GT, use gt as block1 and OCR as block2
    """
    
    if window is None:
        #window = min(len(block1), len(block2))
        window = len(block2)

    aligned_block1 = []
    aligned_block2 = []
    used_indices = set()  # Track used indices in block2

    i = 0
    while i < len(block1):
        line1 = block1[i]
        if threshold is None:
            threshold = len(line1)/2
        
        best_match_index = None
        min_distance = float('inf')

        # Check for the best match within the defined window in block2
        for j in range(max(0, i - window), min(len(block2), i + window + 1)):
            if j not in used_indices:
                line2 = block2[j]
                dist = Levenshtein.distance(re.sub(r'\s+', ' ', line1), re.sub(r'\s+', ' ', line2))
                if dist < min_distance:
                    min_distance = dist
                    best_match_index = j

        # Align if a suitable match is found
        if best_match_index is not None and min_distance <= threshold:
            aligned_block1.append(line1)
            aligned_block2.append(block2[best_match_index])
            used_indices.add(best_match_index)  # Mark this index as used
        else:
            # No match found; add line with blank in block2
            aligned_block1.append(line1)
            aligned_block2.append("")

        i += 1

    # Handle remaining lines in block2 that were not matched
    for j in range(len(block2)):
        if j not in used_indices:
            aligned_block1.append("")
            aligned_block2.append(block2[j])

    return aligned_block1, aligned_block2

if __name__ == "__main__":

    block1 = ("dies ist\n"
              "ein test\n"
              "\n"
              "superduper\n"
              "mega"
              )

    block2 = ("die st\n"
              "ei st\n"
              "megar")

    res1,res2 = align_blocks(block1.splitlines(), block2.splitlines())
    for line1, line2 in zip(res1, res2):
        print(f"{line1:20} | {line2}")
