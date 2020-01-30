def read_solution_from_path(path):
    with open(path, "rt") as f:
        return read_solution(f.readlines())


def read_solution(lines):
    solution = {}
    points = set()
    for line in lines:
        if not line:
            continue
        point_part, centroid_part = line.split('#MAPPING#', 1)[-1].split('->')
        point = tuple([int(x.strip()) for x in point_part.split()])
        centroid = tuple([int(x.strip()) for x in centroid_part.split()])
        if point in points:
            # Skip duplicate points
            print('Point', point, 'is already in the solution; skipping.')
            continue
        points.add(point)
        if not solution.get(centroid):
            solution[centroid] = set()
        solution[centroid].add(point)
    return solution
