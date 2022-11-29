import numpy as np
import statistics as stat

def rectContains(rect, pt, w, h, shrink_factor = 0): 
    """
    Check if a point pt lies inside the bounding rect, considering a
    shrink factor
    """
    [xmin, ymin, width, height] = rect
    cx = xmin + width * 0.5
    cy = ymin + height * 0.5

    xmin = cx - width * 0.5 * (1 - shrink_factor)
    ymin = cy - height * 0.5 * (1 - shrink_factor)
    xmax = cx + width * 0.5 * (1 - shrink_factor)
    ymax = cy + height * 0.5 * (1 - shrink_factor)

    return pt[0] > xmin and pt[0] < xmax and pt[1] > ymin and pt[1] < ymax

def filter_outliers(distances, inliers_sigma=1):
    inliers = []
    mu = stat.mean(distances)
    sigma = stat.stdev(distances)
    for dist in distances:
        if abs(dist - mu)/sigma < inliers_sigma:
            inliers.append(dist)
    return inliers

def get_best_distance(distances, technique="closest"):
    """
    Return one distance value per detection from LiDAR points
    """
    if technique == "closest":
      return min(distances)
    elif technique == "average":
      return stat.mean(distances)
    elif technique == "median":
      return stat.median(sorted(distances))

