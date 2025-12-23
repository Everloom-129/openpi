# think about what metrics to compute, auto ranking the best heads



def calculate_focus_score(image_path):
    """
    Calculate a score indicating how 'focused' the attention is.
    We use the Variance of the heatmap region as a proxy.
    Higher Variance = More focused spots (High peaks vs Low background).
    Lower Variance = Diffuse/Uniform attention.
    """
    if not os.path.exists(image_path):
        return 0.0

    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    # Image layout is 3 columns: [Ext Attn] [Wrist Attn] [Ref Images]
    # We focus on the first column (Exterior Attention) for scoring
    height, width, _ = img.shape
    one_third_width = width // 3

    # Crop the Exterior Attention Heatmap part
    ext_attn_img = img[:, :one_third_width, :]

    # Convert to grayscale to measure intensity variance
    gray = cv2.cvtColor(ext_attn_img, cv2.COLOR_BGR2GRAY)

    # Calculate Variance
    score = np.var(gray)
    return score


def rank_best_heads(reference_timestep=0):
    """
    Scans all layers and heads for a specific timestep and ranks them by focus score.
    """
    print(f"Analyzing heads for timestep {reference_timestep} to find the best ones...")
    scores = []

    # Assume 18 layers and 8 heads (standard for PaliGemma/Pi0)
    for layer in range(18):
        for head in range(8):
            path = get_image_path(reference_timestep, layer, head)
            score = calculate_focus_score(path)
            if score > 0:
                scores.append(((layer, head), score))

    # Sort by score descending (High variance first)
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 Most Focused Heads (Layer, Head):")
    for (layer, head), score in scores[:10]:
        print(f"Layer {layer:02d}, Head {head:02d} | Score: {score:.2f}")

    return [x[0] for x in scores[:10]]
