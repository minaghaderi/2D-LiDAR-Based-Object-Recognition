# Webots LiDAR Object Classification & Dataset Collection

## Project Overview
This project provides a set of Python scripts for collecting 2D LiDAR point cloud datasets using a Webots robot, clustering and classifying objects (such as chairs, tables, and boxes) in real time, and visualizing the results. The system supports dataset collection, live inference with a trained classifier, and easy retraining for new object types.

## Requirements
- **Python 3.8+**
- **Webots** (robotics simulator, https://cyberbotics.com/)
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **joblib**

Install Python dependencies with:
```bash
pip install numpy matplotlib scikit-learn joblib
```

## File Descriptions & How to Run

### 1. `dataset_collector_box.py`
- **Purpose:** Collects LiDAR scans of a box object as the robot orbits it, saving segmented point clouds for dataset creation.
- **How to run:**
  1. Open your Webots world with the robot and the target object (box).
  2. Run the script in the Webots controller environment or from the command line:
     ```bash
     python dataset_collector_box.py
     ```
  3. The script will save `.npy` files in `dataset/box/` as it collects samples.

### 2. `method_a_controller.py`
- **Purpose:** Runs the robot, collects LiDAR data, clusters and classifies objects in real time, and visualizes the results. Also saves snapshots of raw and clustered scans.
- **How to run:**
  1. Ensure `object_classifier.joblib` and `scaler.joblib` are present in the same directory.
  2. Run in the Webots controller environment or from the command line:
     ```bash
     python method_a_controller.py
     ```
  3. The script will display a live plot and save snapshots in the working directory.

### 3. `inference_timer.py`
- **Purpose:** Similar to `method_a_controller.py`, but also measures and prints the inference time for clustering and classification.
- **How to run:**
  1. Ensure `object_classifier.joblib` and `scaler.joblib` are present in the same directory.
  2. Run in the Webots controller environment or from the command line:
     ```bash
     python inference_timer.py
     ```
  3. The script will display a live plot and print inference times to the console.

## How to Collect a Dataset
1. **Set up your Webots world** with the robot and the object you want to collect data for (e.g., box, chair, table).
2. **Edit the save path** in the relevant dataset collector script (e.g., `SAVE_PATH = "dataset/box"`).
3. **Run the script** (e.g., `python dataset_collector_box.py`).
4. **Let the robot orbit and collect samples.** The script will save `.npy` files in the specified dataset directory.
5. **Repeat for each object type** (change the object in the world and the save path).

## How to Retrain the Model (Optional)
1. **Prepare your dataset:** Organize your `.npy` files in folders by class (e.g., `dataset/box/`, `dataset/chair/`, `dataset/table/`).
2. **Extract features:** Write a script to load each `.npy` file, compute features (width, height, area, density, etc.), and save them with labels.
3. **Train a classifier:** Use scikit-learn to train a model (e.g., RandomForest, SVM) on your features. Standardize features with `StandardScaler`.
4. **Save the model and scaler:**
   ```python
   from joblib import dump
   dump(clf, 'object_classifier.joblib')
   dump(scaler, 'scaler.joblib')
   ```
5. **Replace the old model files** in your project directory with the new ones.

## Notes
- All scripts are designed to be run as Webots robot controllers.
- The `.npy` files are NumPy arrays of 2D point clouds.
- The classifier expects features similar to those used in the provided scripts.

## License
MIT License (or specify your own) 