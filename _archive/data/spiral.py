import torch
from torch.utils.data import Dataset


class SpiralDataset(Dataset):
    def __init__(self, num_samples=1_000, num_extra_features=0, noise=0.2):
        self.num_samples = num_samples
        self.num_extra_features = num_extra_features

        # Calculate samples per class based on total samples
        num_classes = 2  # Hardcoded to 2 for this version
        samples_per_class = num_samples // num_classes

        # Generate spiral data (MODIFIED for longer arms)
        features_list = []
        labels_list = []
        for j in range(num_classes):
            r = torch.linspace(0.0, 2, samples_per_class)  # Extend radius to 2
            t = torch.linspace(
                j * 8,
                (j + 1) * 8,
                samples_per_class,  # Double the angle range
            ) + noise * torch.randn(samples_per_class)

            x = r * torch.sin(t)
            y = r * torch.cos(t)

            features_list.append(torch.stack([x, y], dim=1))
            # Adjusted label assignment for -1 and 1
            labels_list.append(
                torch.full((samples_per_class,), 0 if j == 0 else 1, dtype=torch.float)
            )

        # Optionally add extra features
        if num_extra_features > 0:
            extra_data = torch.randn(self.num_samples, num_extra_features)
            for i in range(len(features_list)):
                features_list[i] = torch.cat(
                    [
                        features_list[i],
                        extra_data[i * samples_per_class : (i + 1) * samples_per_class],
                    ],
                    dim=1,
                )

        self.features = torch.cat(features_list, dim=0).float()
        self.labels = torch.cat(labels_list, dim=0).unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
