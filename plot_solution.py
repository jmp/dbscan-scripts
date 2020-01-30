import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from os import system
from sys import argv

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["font.sans-serif"] = "Times New Roman"
mpl.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"

AXIS_PARAMS = {"fontname": "Times New Roman"}

POINT_COLOR = "#666666"
NOISE_COLOR = "#ff0000"
GROUND_TRUTH_COLOR = "#000000"
RESULT_COLOR = "#0088ff"
CONVEX_HULL_COLOR = "#333333"


def convex_hull(points):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def distance_squared(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def distance_non_squared(p, q):
    from math import sqrt
    return sqrt(distance_squared(p, q))


def closest_index(p, centroids):
    closest_idx = -1
    closest_distance = None
    for i, centroid in enumerate(centroids):
        distance = WEIGHTS[i] * distance_squared(p, centroid)
        if closest_distance is None or distance < closest_distance:
            closest_idx = i
            closest_distance = distance
    return closest_idx


# Calculate partitions
def calculate_partitions(solution):
    return [points for points in solution.values()]


def plot_solution(solution, points):
    partitions = calculate_partitions(solution)
    convex_hulls = [np.array(convex_hull(p_points)) for p_points in partitions]
    solution_points = [p for cluster in solution.values() for p in cluster]

    noise_points = set(points).difference(set(solution_points))

    x = [x for x, _ in points]
    y = [y for _, y in points]

    # Render plot
    plt.figure()

    # Draw convex hulls
    for hull in convex_hulls:
        t = plt.Polygon(hull, color=CONVEX_HULL_COLOR, fill=False)
        plt.gca().add_patch(t)

    # Draw points
    plt.scatter(*zip(*points), c=POINT_COLOR, s=1, marker='.', edgecolors=None)

    # Draw possible noise points
    if noise_points:
        plt.scatter(*zip(*noise_points), c=NOISE_COLOR, s=3, marker='.', edgecolors=None)

    plt.axis('off')

    # Draw centroids
    for centroid in solution.keys():
        plt.scatter(*zip(*[centroid]), c=GROUND_TRUTH_COLOR, s=30)

    #plt.xlim(0, 2500)
    #plt.ylim(0, 2500)
    plt.axes().set_aspect("equal")
    #plt.margins(0.1, 0.05)
    #plt.xticks(np.arange(0, 10001, 2000))
    #plt.yticks(np.arange(0, 9001, 2000))
    plt.tight_layout()
    plt.show()
    # plt.savefig("filename.svg", bbox_inches='tight', pad_inches=0)
