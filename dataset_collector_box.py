#dataset creation:
# === Webots box Dataset Collector with Full 360 Orbit and Varying Distance ===
# This script collects LIDAR point cloud data from a Webots robot as it orbits a box, saving segmented object scans for dataset creation.
# The robot follows a circular path with varying radius, and the code clusters, segments, and saves the largest detected object periodically.
# The script also visualizes the robot's path and detected objects in real time.
#
# Key features:
# - Adaptive clustering of LIDAR points
# - Wall and door detection
# - Periodic saving of object scans
# - Real-time matplotlib visualization
#
# === Imports ===
from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

# === Helper Functions ===
def get_heading(compass):
    n = compass.getValues()
    return -math.atan2(n[0], n[1])

def adaptive_eps(points, base_eps=0.15):
    if len(points) < 2:
        return base_eps
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)
    mean_dist = np.mean(distances[:, 1])
    return np.clip(base_eps * (1 + mean_dist * 2), 0.08, 0.32)

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

def detect_door_frames(wall_segments, threshold_min=1.1, threshold_max=2, angle_tol_deg=10):
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

def filter_close_door_centers(door_centers, min_dist=1.0):
    if len(door_centers) == 0:
        return np.array([])
    db = DBSCAN(eps=min_dist, min_samples=1).fit(door_centers)
    labels = db.labels_
    return np.array([np.mean(door_centers[labels == lbl], axis=0) for lbl in set(labels)])

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
    features, ids = [], []
    for obj_id in final_objects:
        if obj_id in wall_ids:
            continue
        pts = final_objects[obj_id]
        width, height = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        area = width * height
        if area == 0:
            continue
        features.append([width, height, width/height if height else 0, len(pts)/area])
        ids.append(obj_id)
    classified = {}
    for obj_id in ids:
        classified[obj_id] = ""
    return final_objects, classified, wall_ids, doors

# === Setup ===
# Directory to save collected scans
SAVE_PATH = "dataset/box"
os.makedirs(SAVE_PATH, exist_ok=True)
# Counter for saved samples
saved_count = 76
# Maximum number of samples to collect
max_samples = 200
# Enable/disable saving
saving_enabled = True

# Initialize Webots robot and devices
robot = Robot()
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

# Robot kinematics parameters
wheel_radius = 0.033
wheel_base = 0.16
x = y = theta = prev_l = prev_r = 0.0
# List to store all global LIDAR points
global_points = []
# List to store robot's path
robot_path = []

# Motion and clustering parameters
base_speed = 3.0
mod_factor = 0.6
clustering_interval = 2.0
last_cluster_time = -1.0

# Setup matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

start_time = robot.getTime()
final_objects, classified_objects, wall_ids, door_centers = {}, {}, [], []

# === Main Loop: Orbiting pattern with varying radius ===
angle = 0
radius = random.uniform(0.5, 1.2)
angular_velocity = 0.25  # radians per second

while robot.step(timestep) != -1:
    # Stop after 400 seconds (safety cap)
    if robot.getTime() - start_time > 400:  # hard cap
        break

    # Update orbit angle and radius
    angle += angular_velocity * timestep / 1000.0
    radius += random.uniform(-0.01, 0.01)
    radius = np.clip(radius, 0.5, 1.2)
    # Calculate wheel speeds for orbiting
    v_left = base_speed * (1 - mod_factor * math.sin(angle))
    v_right = base_speed * (1 + mod_factor * math.sin(angle))
    left_motor.setVelocity(v_left)
    right_motor.setVelocity(v_right)

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
    distances = distances[valid][::10]
    angle_sub = angles[valid][::10]
    # Convert scan to global coordinates
    lx = distances * np.cos(angle_sub)
    ly = distances * np.sin(angle_sub)
    gx = x + lx * np.cos(theta) - ly * np.sin(theta)
    gy = y + lx * np.sin(theta) + ly * np.cos(theta)
    scan_points = np.stack((gx, gy), axis=1)
    global_points.extend(scan_points)

    # Periodically cluster and save largest object
    if saving_enabled and robot.getTime() - last_cluster_time >= clustering_interval:
        last_cluster_time = robot.getTime()

        # Randomly remove 10% of the global points before next scan (to avoid memory bloat)
        if len(global_points) > 20:
            global_points = list(global_points)
            num_to_remove = int(len(global_points) * 0.1)
            remove_indices = np.random.choice(len(global_points), size=num_to_remove, replace=False)
            global_points = [pt for i, pt in enumerate(global_points) if i not in remove_indices]

        # Run clustering and segmentation
        final_objects, classified_objects, wall_ids, door_centers = run_clustering_and_classification(np.array(global_points))

        # Find the largest non-wall object
        largest_obj = None
        max_pts = 0
        for obj_id, pts in final_objects.items():
            if obj_id in wall_ids or len(pts) < 10:
                continue
            if len(pts) > max_pts:
                largest_obj = (obj_id, pts)
                max_pts = len(pts)

        # Save the largest object as a .npy file
        if largest_obj and saved_count < max_samples:
            np.save(f"{SAVE_PATH}/scan_{saved_count:03d}.npy", largest_obj[1])
            print(f"[INFO] Saved sample #{saved_count} with {len(largest_obj[1])} points.")
            saved_count += 1

        # Stop saving after reaching max_samples
        if saved_count >= max_samples:
            saving_enabled = False
            print("[INFO] Dataset collection completed. Robot continues orbiting.")

    # === Visualization ===
    ax.cla()
    ax.set_title(f"Dataset Collection - Saved: {saved_count}")
    if len(global_points) > 0:
        global_np = np.array(global_points)
        ax.scatter(global_np[:, 0], global_np[:, 1], s=2, color='gray')
    for pts in final_objects.values():
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color='cyan')
    plt.pause(0.01)

# Stop the robot and save the final map
left_motor.setVelocity(0)
right_motor.setVelocity(0)
plt.ioff()
plt.savefig("box_dataset_final_map.png")
plt.show()