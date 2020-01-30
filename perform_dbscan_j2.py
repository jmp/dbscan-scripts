import matplotlib.pyplot as plt
import numpy as np
import csi
import read_mapping
import plot_solution

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# Load points
points = []
with open("j2.txt", "rt") as f:
    for line in f.readlines():
        x, y = line.split()[:2]
        points.append((int(x), int(y)))

# Load ground truth solution
solution_gt = read_mapping.read_solution_from_path('mapping-j2-gt.txt')


# #############################################################################
# Generate sample data
X = np.asarray(points)
# X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
best_csi = 0
best_eps = 0
best_ci = 100000000
best_ci_eps = 0
labels = None
core_samples_mask = None


def perform_dbscan(eps):
    global labels
    global core_samples_mask

    db = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    points = set()
    solution = {}
    for i, label in enumerate(labels):
        point = tuple(X[i])
        if label == -1:
            # print('noise', point)
            continue
        if point in points:
            # print("skip", point)
            continue
        if not solution.get(label):
            solution[label] = set()
        solution[label].add(point)


    def calculate_centroid(points):
        average_x = 0
        average_y = 0
        for point in points:
            average_x += point[0]
            average_y += point[1]
        return average_x / len(points), average_y / len(points)


    tmp_solution = {}
    for label, points in solution.items():
        centroid = calculate_centroid(points)
        #print('CENTROID:', centroid)
        tmp_solution[centroid] = points

    solution = tmp_solution

    return solution


for eps in range(1, 200):
    solution = perform_dbscan(eps)

    curr_csi = csi.csi(solution_gt, solution)
    if curr_csi >= best_csi:
        best_csi = curr_csi
        best_eps = eps

    curr_ci = csi.centroid_index(solution_gt.keys(), solution.keys())
    if curr_ci <= best_ci:
        best_ci = curr_ci
        best_ci_eps = eps

    print('CSI:', curr_ci, 'BEST:', best_csi, 'EPS:', best_eps)
    print('CI:', curr_ci, 'BEST:', best_ci, 'EPS:', best_ci_eps)


perform_dbscan(best_eps)

# print(solution)
print('NUM_CLUSTERS', len(solution.keys()))

plot_solution.plot_solution(solution, points)

#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     plt.axis('off')
#     plt.axes().set_aspect("equal")
#
#     #xy = X[class_member_mask & core_samples_mask]
#     #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#     #         markeredgecolor=None, markersize=1)
#
#     xy = X[class_member_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor=None, markersize=1)
#
# plt.show()

print('BEST CSI:', best_csi, 'EPS:', best_eps)
print('BEST CI:', best_ci, 'EPS:', best_ci_eps)

# # #############################################################################
# # Plot result
# #
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     plt.axis('off')
#     plt.axes().set_aspect("equal")
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor=None, markersize=1)
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor=None, markersize=1)
#
# plt.show()