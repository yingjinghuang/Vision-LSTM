import os

# basic paths
project_folder = "/home/huangyj/urban_village"
data_folder = os.path.join(project_folder, "data")
detail_folder = os.path.join(data_folder, "grids_all_500m")

tensorboard_folder = os.path.join(project_folder, "tensorboard")
weights_folder = os.path.join(project_folder, "weights")

# paths
sv_final_csv_path = os.path.join(detail_folder, "sv_final.csv")
sv_feature_csv_path = os.path.join(detail_folder, "sv_features.csv")
taxi_final_csv_path = os.path.join(detail_folder, "taxi_final_340.csv")
labels_csv_path = os.path.join(detail_folder, "label_final.csv")
image_dir = "/shared_ssd/huangyj/resized/remote_intersect_500m_google_resize"
sv_dir = "/shared_ssd/huangyj/resized"

# sv_gid_selected_path = os.path.join(detail_folder, "sv_topn_path.csv")

test_dir = os.path.join(project_folder, "test_output")

model_data_path = os.path.join(detail_folder, "data_model.pkl")
model_data_path2 = os.path.join(detail_folder, "data_model4single_sv5.pkl")