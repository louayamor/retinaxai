from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Ordinal-aware cross-entropy loss for DR grading.

    Standard cross-entropy treats all misclassifications equally. For ordinal data
    like DR grades (0→No DR, 1→Mild, 2→Moderate, 3→Severe, 4→Proliferative DR),
    predicting grade 4 when true label is 0 is worse than predicting grade 1.

    This loss adds a distance penalty proportional to |predicted_class - true_class|.

    Reference: Cheng et al., "Ordinal Regression with Neural Networks for Medical Diagnosis"
    """

    def __init__(
        self,
        num_classes: int = 5,
        distance_weight: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Args:
            num_classes: Number of ordinal classes (default 5 for DR grades 0-4)
            distance_weight: Weight for the distance penalty (default 0.1)
            class_weights: Per-class weight tensor of shape (num_classes,)
        """
        super().__init__()
        self.num_classes = num_classes
        self.distance_weight = distance_weight
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal cross-entropy loss with distance penalty.

        Args:
            logits: Tensor of shape (batch_size, num_classes) - raw model outputs
            targets: Tensor of shape (batch_size,) - integer class labels [0, num_classes-1]

        Returns:
            Scalar loss tensor
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)

        probs = F.softmax(logits, dim=1)
        predicted_classes = probs.argmax(dim=1)

        distance = torch.abs(predicted_classes.float() - targets.float())
        distance_penalty = distance.mean()

        return ce_loss + self.distance_weight * distance_penalty


class CoralLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) loss for ordinal regression.

    Treats K-class ordinal problem as K-1 binary classification tasks.
    Each binary task predicts P(y > k) for k in [0, K-2].

    Requires model to output K-1 binary logits instead of K class logits.

    Reference: Niu et al., "Ordinal Regression with Multiple Output CNN for Age Estimation", CVPR 2016.
    """

    def __init__(self, num_classes: int = 5):
        """
        Args:
            num_classes: Number of ordinal classes (default 5 for DR grades 0-4)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_binary_tasks = num_classes - 1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL loss.

        Args:
            logits: Tensor of shape (batch_size, num_classes-1) - binary logits for P(y > k)
            targets: Tensor of shape (batch_size,) - integer class labels [0, num_classes-1]

        Returns:
            Scalar loss tensor
        """
        batch_size = logits.size(0)

        labels = torch.zeros(
            batch_size, self.num_binary_tasks, dtype=torch.float32, device=logits.device
        )

        for k in range(self.num_binary_tasks):
            labels[:, k] = (targets > k).float()

        loss = nn.BCEWithLogitsLoss()(logits, labels)

        return loss
