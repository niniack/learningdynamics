import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    def __init__(self, num_samples=1_000, num_extra_features=1):
        self.num_classes = 2
        self.num_samples = num_samples
        self.num_extra_features = num_extra_features

        # Number of samples per class
        samples_per_class = self.num_samples // self.num_classes

        # Define ranges for each class
        class_means = [[-1, -1], [1, 1]]

        features_list = []
        labels_list = []

        for cl in range(self.num_classes):
            x1_mean, x2_mean = class_means[cl][0], class_means[cl][1]

            # Efficiently sample from uniform distributions
            x1_data = torch.FloatTensor(samples_per_class).normal_(mean=x1_mean)
            x2_data = torch.FloatTensor(samples_per_class).normal_(mean=x2_mean)

            # Create extra_data only if num_extra_features > 0
            if self.num_extra_features > 0:
                extra_data = torch.FloatTensor(
                    samples_per_class, num_extra_features
                ).normal_(mean=0, std=0.01)
                features_list.append(
                    torch.stack(
                        [x1_data.unsqueeze(1), x2_data.unsqueeze(1), extra_data], dim=1
                    ).squeeze(-1)
                )
            else:
                features_list.append(torch.stack([x1_data, x2_data], dim=1))

            lbl = torch.full((samples_per_class, 1), 1.0 if cl == 0 else -1.0)
            labels_list.append(lbl)

        self.features = torch.cat(features_list, dim=0).float()
        self.labels = torch.cat(labels_list, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input = self.features[idx]
        label = self.labels[idx]

        return input, label
