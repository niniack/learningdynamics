import torch


def mean_classifier(features, labels, x=[0.2, 0.8]):
    class_a = features[torch.where(labels == -1)[0]]
    class_b = features[torch.where(labels == 1)[0]]

    p1 = torch.mean(class_a, dim=0, keepdim=False)
    p2 = torch.mean(class_b, dim=0, keepdim=False)

    px = [p1[0], p2[0]]
    py = [p1[1], p2[1]]

    # Midpoint of p1 and p2
    midpoint = (p1 + p2) / 2.0

    # Direction vector of the bisector (perpendicular to p2 - p1)
    slope = torch.tensor([(p2[1] - p1[1]) / (p2[0] - p1[0])])
    perp_slope = -1 / slope

    intercept = midpoint[1] - perp_slope * midpoint[0]

    endpoint2 = [
        float(perp_slope * x[0] + intercept),
        float(perp_slope * x[1] + intercept),
    ]

    return x, endpoint2
