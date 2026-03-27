import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HARVideoDataset(Dataset):
    def __init__(self, video_paths, labels, max_frames=200):
        self.video_paths = video_paths
        self.labels = labels
        self.max_frames = max_frames

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

        self.cached_videos = self._cache_videos()

    def _cache_videos(self):
        cache = {}
        for idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)

            cap.release()

            # Adjust frame count
            if len(frames) > self.max_frames:
                frames = frames[:self.max_frames]
            else:
                pad = torch.zeros_like(frames[0])
                frames.extend([pad] * (self.max_frames - len(frames)))

            cache[idx] = torch.stack(frames)

        return cache

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return self.cached_videos[idx], self.labels[idx]
