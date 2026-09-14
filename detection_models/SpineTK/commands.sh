# Fine-tune Detectron using CSXA data
python run.py configs/csxa.yaml

# Do inference on the entire test split, calculate keypoint errors (pixel error)
# and save all results to a .csv
python compute_keypoint_error.py configs/csxa.yaml --split test --csv keypoint_errors.csv

# Randomly select *limit* images from the test split
# do inference and pliot the ground truth keypoints and the predictions together
# save the images to the prediction_plots folder
python plot_predictions.py configs/csxa.yaml --split test --limit 5 --seed 42 --output-dir prediction_plots

