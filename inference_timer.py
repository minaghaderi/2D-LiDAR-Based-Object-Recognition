#measuring the inference time
# === Webots LiDAR Inference Timer and Object Classifier ===
# This script runs a Webots robot, collects LIDAR point cloud data, clusters and classifies objects in real time,
# and measures the inference time for clustering and classification. It visualizes the robot's path, detected objects,
# and classification results live using matplotlib.
#
# Key features:
# - Loads a trained classifier and scaler for object recognition
# - Adaptive clustering and segmentation of LIDAR points
# - Wall and door detection
# - Real-time classification and timing of inference
# - Live matplotlib visualization of results
#
# === Imports ===
from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from matplotlib import cm
from joblib import load
import time
# === Load Classifier ===
CLASSES = ["chair", "table", "box"]
clf = load("object_classifier.joblib")    # the actual trained model
scaler = load("scaler.joblib")            # the standard scaler used during training

# === Helper Functions ===
# Get the robot's heading (orientation) from the compass sensor
def get_heading(compass):
    n = compass.getValues()
    return -math.atan2(n[0], n[1])

# Compute an adaptive DBSCAN epsilon based on local point density
def adaptive_eps(points, base_eps=0.15):
    if len(points) < 2:
        return base_eps
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)
    mean_dist = np.mean(distances[:, 1])
    return np.clip(base_eps * (1 + mean_dist * 2), 0.08, 0.32)

# Compute the curvature at a point given its neighbors
def compute_curvature(p_minus, p, p_plus):
    a = np.linalg.norm(p - p_minus)
    b = np.linalg.norm(p - p_plus)
    c = np.linalg.norm(p_plus - p_minus)
    if a * b * c == 0:
        return 0
    s = (a + b + c) / 2
    area = math.sqrt(abs(s * (s - a) * (s - b) * (s - c)))
    return (4 * area) / (a * b * c)

# Split a wall segment by detecting sharp corners (horizontal sorting)
def split_wall_by_corner(points, angle_threshold_deg=35):
    if len(points) < 3:
        return [points]
    points = points[np.argsort(points[:, 0])]
    segments = []
    current = [points[0]]
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi
        current.append(points[i])
        if angle > angle_threshold_deg:
            segments.append(np.array(current))
            current = []
    current.append(points[-1])
    if current:
        segments.append(np.array(current))
    return segments

# Split a wall segment by detecting sharp corners (vertical sorting)
def split_wall_by_corner_vertical(points, angle_threshold_deg=35):
    if len(points) < 3:
        return [points]
    points = points[np.argsort(points[:, 1])]
    segments = []
    current = [points[0]]
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi
        current.append(points[i])
        if angle > angle_threshold_deg:
            segments.append(np.array(current))
            current = []
    current.append(points[-1])
    if current:
        segments.append(np.array(current))
    return segments

# Detect door frames by finding pairs of parallel wall segments at door-like distances
def detect_door_frames(wall_segments, threshold_min=1.3, threshold_max=2, angle_tol_deg=10):
    door_centers = []
    wall_list = list(wall_segments.items())
    for i in range(len(wall_list)):
        for j in range(i+1, len(wall_list)):
            pts1, pts2 = wall_list[i][1], wall_list[j][1]
            if np.linalg.norm(pts1[-1] - pts1[0]) < 0.3 or np.linalg.norm(pts2[-1] - pts2[0]) < 0.3:
                continue
            v1 = pts1[-1] - pts1[0]
            v2 = pts2[-1] - pts2[0]
            v1 /= np.linalg.norm(v1) + 1e-6
            v2 /= np.linalg.norm(v2) + 1e-6
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi
            if angle > angle_tol_deg:
                continue
            min_dist = np.min([np.linalg.norm(p1 - p2) for p1 in pts1 for p2 in pts2])
            if threshold_min <= min_dist <= threshold_max:
                pair = min(((p1, p2) for p1 in pts1 for p2 in pts2), key=lambda x: np.linalg.norm(x[0] - x[1]))
                center = (pair[0] + pair[1]) / 2
                door_centers.append(center)
    return np.array(door_centers)

# Merge close door center detections using DBSCAN
def filter_close_door_centers(door_centers, min_dist=1.0):
    if len(door_centers) == 0:
        return np.array([])
    db = DBSCAN(eps=min_dist, min_samples=1).fit(door_centers)
    labels = db.labels_
    return np.array([np.mean(door_centers[labels == lbl], axis=0) for lbl in set(labels)])

# Cluster points, merge clusters, split walls, and classify objects
def run_clustering_and_classification(points):
    if len(points) < 10:
        return {}, {}, [], []

    eps = adaptive_eps(points)
    db = DBSCAN(eps=eps, min_samples=10).fit(points)
    labels = db.labels_

    clusters, centroids = [], []
    for label in set(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        if len(cluster) < 20:
            continue
        clusters.append(cluster)
        centroids.append(np.mean(cluster, axis=0))

    if not centroids:
        return {}, {}, [], []

    centroids = np.array(centroids)
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1.3).fit(centroids)
    super_labels = agg.labels_

    merged_objects = {}
    for idx, lbl in enumerate(super_labels):
        if lbl not in merged_objects:
            merged_objects[lbl] = clusters[idx]
        else:
            merged_objects[lbl] = np.vstack((merged_objects[lbl], clusters[idx]))

    final_objects, wall_ids = {}, []
    for obj_id, pts in merged_objects.items():
        if len(pts) < 8:
            continue
        width, height = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        area = width * height
        ratio = max(width / height, height / width)
        final_objects[obj_id] = pts
        if area > 2.0 or ratio > 4:
            wall_ids.append(obj_id)

    new_final, new_wall_ids, counter = {}, [], 0
    for obj_id, pts in final_objects.items():
        if obj_id in wall_ids:
            for seg in split_wall_by_corner(pts) + split_wall_by_corner_vertical(pts):
                if len(seg) >= 5:
                    new_final[counter] = seg
                    new_wall_ids.append(counter)
                    counter += 1
        else:
            new_final[counter] = pts
            counter += 1

    final_objects = new_final
    wall_ids = new_wall_ids
    doors = filter_close_door_centers(detect_door_frames({i: final_objects[i] for i in wall_ids}))

    classified = {}
    for obj_id in final_objects:
        if obj_id in wall_ids:
            continue
        pts = final_objects[obj_id]
        width = np.ptp(pts[:, 0])
        height = np.ptp(pts[:, 1])
        area = width * height if height else 1e-6
        density = len(pts) / area if area > 0 else 0
        point_count = len(pts)
        aspect_ratio = width / height if height else 0
        std_x = np.std(pts[:, 0])
        std_y = np.std(pts[:, 1])
        std_ratio = std_x / std_y if std_y else 0

        feats = [[
        width, height, area, density,
        aspect_ratio, point_count,
        std_x, std_y, std_ratio
        ]]
        scaled_feats = scaler.transform(feats)   # ✅ apply scaling
        label_index = clf.predict(scaled_feats)[0]
        classified[obj_id] = CLASSES[label_index]

    return final_objects, classified, wall_ids, doors

# Update the live matplotlib plot with current scan, path, objects, and labels
def update_live_plot(global_points, robot_path, final_objects, classified_objects, door_centers, t):
    plt.clf()
    ax = plt.gca()
    ax.set_title(f"Live LiDAR Classification View - Time: {t:.3f} sec")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')
    ax.grid(True)

    # Rotate + mirror global scan points
    global_np = np.array(global_points)
    if len(global_np) > 0:
        rotated_np = rotate_and_mirror(global_np)
        ax.scatter(rotated_np[:, 0], rotated_np[:, 1], s=2, c='lightgray')

    # Rotate + mirror robot path
    if len(robot_path) > 1:
        path_np = rotate_and_mirror(np.array(robot_path))
        ax.plot(path_np[:, 0], path_np[:, 1], 'b-', linewidth=1)

    # Plot object clusters and bounding boxes
    for i, (obj_id, pts) in enumerate(final_objects.items()):
        color = 'cyan'
        pts_rot = rotate_and_mirror(pts)
        ax.scatter(pts_rot[:, 0], pts_rot[:, 1], s=6, color=color)

        xmin, xmax = np.min(pts_rot[:, 0]), np.max(pts_rot[:, 0])
        ymin, ymax = np.min(pts_rot[:, 1]), np.max(pts_rot[:, 1])
        ax.plot([xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color='red', linewidth=2)

        label = classified_objects.get(obj_id, "")
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        ax.text(cx, cy + 0.1, label, fontsize=8, color='blue', ha='center')

    # Plot door frame centers
    if len(door_centers) > 0:
        door_rot = rotate_and_mirror(np.array(door_centers))
        for center in door_rot:
            ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2)
            ax.text(center[0], center[1] + 0.1, 'doorframe', fontsize=8, color='black', ha='center')

    # Axis limits
    if len(global_np) > 0:
        rotated_np = rotate_and_mirror(global_np)
        ax.set_xlim(np.min(rotated_np[:, 0]) - 1, np.max(rotated_np[:, 0]) + 1)
        ax.set_ylim(np.min(rotated_np[:, 1]) - 1, np.max(rotated_np[:, 1]) + 1)

    plt.pause(0.01)

# Rotate 90° clockwise and then mirror horizontally (flip X)
def rotate_and_mirror(points):
    """ Rotate 90° clockwise and then mirror horizontally (flip X) """
    rotated = np.column_stack((points[:, 1], -points[:, 0]))  # 90° CW
    mirrored = np.column_stack((-rotated[:, 0], rotated[:, 1]))  # Flip X
    return mirrored

# === Webots Controller Logic ===
# Initialize robot and devices
robot = Robot()
start_time = robot.getTime()
t = robot.getTime() - start_time
timestep = int(robot.getBasicTimeStep())
lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)
lidar.enablePointCloud()
res = lidar.getHorizontalResolution()
fov = lidar.getFov()
angles = np.linspace(-fov/2, fov/2, res)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_encoder = robot.getDevice("left wheel sensor")
right_encoder = robot.getDevice("right wheel sensor")
left_encoder.enable(timestep)
right_encoder.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Robot kinematics and state variables
wheel_radius = 0.033
wheel_base = 0.16
x = y = theta = prev_l = prev_r = 0.0
# List to store all global LIDAR points
global_points = []
# List to store robot's path
robot_path = []

# Motion and clustering parameters
base_speed = 1.5
mod_factor = 1.2
max_time = 15
last_cluster_time = -1.0
clustering_interval = 2.0

# Setup matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

start_time = robot.getTime()
final_objects, classified_objects, wall_ids, door_centers = {}, {}, [], []

# === Main Loop: Move robot, collect scans, classify, and visualize ===
while robot.step(timestep) != -1:
    t = robot.getTime() - start_time
    if t > max_time:
        break

    # Update wheel speeds for oscillating motion
    mod = abs(math.sin(t / 2))
    left_motor.setVelocity(base_speed * (1 - mod_factor * mod))
    right_motor.setVelocity(base_speed * (1 + mod_factor * mod))

    # Odometry update
    lpos = left_encoder.getValue()
    rpos = right_encoder.getValue()
    dl = wheel_radius * (lpos - prev_l)
    dr = wheel_radius * (rpos - prev_r)
    prev_l, prev_r = lpos, rpos
    dc = (dl + dr) / 2.0
    dtheta = (dr - dl) / wheel_base
    theta += dtheta
    x += dc * math.cos(theta)
    y += dc * math.sin(theta)
    theta = get_heading(compass)
    robot_path.append((x, y))

    # Get and filter LIDAR scan
    distances = np.array(lidar.getRangeImage())
    valid = np.isfinite(distances) & (distances > 0.1) & (distances < 3.5)
    distances = distances[valid][::1]
    angle_sub = angles[valid][::1]
    # Convert scan to global coordinates
    lx = distances * np.cos(angle_sub)
    ly = distances * np.sin(angle_sub)
    gx = x + lx * np.cos(theta) - ly * np.sin(theta)
    gy = y + lx * np.sin(theta) + ly * np.cos(theta)
    scan_points = np.stack((gx, gy), axis=1)
    global_points.extend(scan_points)

    # Periodically cluster, classify, and measure inference time
    if t - last_cluster_time >= clustering_interval:
        
        time_start = time.time()

        final_objects, classified_objects, wall_ids, door_centers = run_clustering_and_classification(np.array(global_points))

        time_end = time.time()
        elapsed_ms = (time_end - time_start) * 1000  # milliseconds
        print(f"Inference Time for Classification + Door Detection: {elapsed_ms:.2f} ms")

        last_cluster_time = t

    update_live_plot(global_points, robot_path, final_objects, classified_objects, door_centers, t)

# Stop the robot and save the final map
left_motor.setVelocity(0)
right_motor.setVelocity(0)
plt.ioff()
plt.savefig("final_live_map_detected.png")
plt.show()

