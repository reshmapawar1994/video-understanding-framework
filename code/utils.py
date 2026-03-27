import os

def load_dataset_paths(root_path):
    classes = sorted(os.listdir(root_path))
    paths, labels = [], []

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(root_path, cls)
        if os.path.isdir(cls_path):
            for file in os.listdir(cls_path):
                if file.endswith((".mp4", ".avi")):
                    paths.append(os.path.join(cls_path, file))
                    labels.append(idx)

    return paths, labels, classes
