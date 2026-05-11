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
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            num_classes: Number of ordinal classes (default 5 for DR grades 0-4)
            distance_weight: Weight for the distance penalty (default 0.1)
            class_weights: Per-class weight tensor of shape (num_classes,)
            label_smoothing: Label smoothing factor (default 0.0, no smoothing)
        """
        super().__init__()
        self.num_classes = num_classes
        self.distance_weight = distance_weight
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal cross-entropy loss with distance penalty.

        Uses expected class distance (differentiable) instead of argmax (non-differentiable).

        Args:
            logits: Tensor of shape (batch_size, num_classes) - raw model outputs
            targets: Tensor of shape (batch_size,) - integer class labels [0, num_classes-1]

        Returns:
            Scalar loss tensor
        """
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(
                logits, self.label_smoothing / self.num_classes
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            ce_loss = (-smooth_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)

        probs = F.softmax(logits, dim=1)

        class_indices = torch.arange(
            self.num_classes, device=logits.device, dtype=torch.float32
        )
        expected_class = torch.sum(probs * class_indices, dim=1)

        distance = torch.abs(expected_class - targets.float())
        distance_penalty = distance.mean()

        return ce_loss + self.distance_weight * distance_penalty


class FocalOrdinalLoss(nn.Module):
    """
    Focal loss + ordinal distance penalty for DR grading.

    Focal loss down-weights easy examples (high confidence correct predictions)
    and focuses training on hard misclassifications. Combined with ordinal
    distance penalty, this reduces the model's tendency to default to the
    majority class (Grade 0) when uncertain.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        num_classes: int = 5,
        distance_weight: float = 0.1,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            num_classes: Number of ordinal classes (default 5 for DR grades 0-4)
            distance_weight: Weight for the distance penalty (default 0.1)
            gamma: Focal focusing parameter (default 2.0)
            class_weights: Per-class weight tensor of shape (num_classes,)
            label_smoothing: Label smoothing factor (default 0.0)
        """
        super().__init__()
        self.num_classes = num_classes
        self.distance_weight = distance_weight
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal ordinal loss with distance penalty.

        Uses expected class distance (differentiable) instead of argmax (non-differentiable).

        Args:
            logits: Tensor of shape (batch_size, num_classes) - raw model outputs
            targets: Tensor of shape (batch_size,) - integer class labels [0, num_classes-1]

        Returns:
            Scalar loss tensor
        """
        probs = F.softmax(logits, dim=1)

        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(
                logits, self.label_smoothing / self.num_classes
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            ce_loss = (-smooth_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(
                logits, targets, weight=self.class_weights, reduction="none"
            )

        if self.label_smoothing == 0:
            probs_target = probs[torch.arange(logits.size(0)), targets]
            focal_weight = (1 - probs_target) ** self.gamma
            focal_loss = (focal_weight * ce_loss).mean()
        else:
            focal_loss = ce_loss.mean()

        class_indices = torch.arange(
            self.num_classes, device=logits.device, dtype=torch.float32
        )
        expected_class = torch.sum(probs * class_indices, dim=1)

        distance = torch.abs(expected_class - targets.float())
        distance_penalty = distance.mean()

        return focal_loss + self.distance_weight * distance_penalty


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
