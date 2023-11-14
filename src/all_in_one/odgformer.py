from typing import Tuple
import torch.nn.functional as F
import torch
from torch import Tensor
from sklearn.metrics import precision_recall_curve
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from sklearn.metrics import auc as auc_fn

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)
        

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

    def get_log_prob(self, x, edge_index):
        x = self.forward(x, edge_index)
        return F.log_softmax(x)
        


class OSEM(AllInOne):
    def __init__(
        self,
        inference_steps: int,
        lambda_s: float,
        lambda_z: float,
        ema_weight: float = 1.0,
        use_inlier_latent: bool = True,
    ):
        super().__init__()
        self.inference_steps = inference_steps
        self.lambda_s = lambda_s
        self.lambda_z = lambda_z
        self.ema_weight = ema_weight
        self.use_inlier_latent = use_inlier_latent

    def cosine(self, X, Y):
        return F.normalize(X, dim=-1) @ F.normalize(Y, dim=-1).T

    def get_logits(self, prototypes: Tensor, query_features: Tensor) -> Tensor:
        return self.cosine(query_features, prototypes)  # [query_size, num_classes]

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Metric dic
        num_classes = support_labels.unique().size(0)
        one_hot_labels = F.one_hot(
            support_labels, num_classes
        )  # [support_size, num_classes]
        support_size = support_features.size(0)

        prototypes = compute_prototypes(
            support_features, support_labels
        )  # [num_classes, feature_dim]
        soft_assignements = (1 / num_classes) * torch.ones(
            query_features.size(0), num_classes
        )  # [query_size, num_classes]
        inlier_scores = 0.5 * torch.ones((query_features.size(0), 1))

        # To measure the distance to the true prototype
        outliers = kwargs["outliers"].bool()
        inliers = ~outliers
        true_prototypes = compute_prototypes(
            torch.cat((support_features, query_features[inliers])),
            torch.cat((support_labels, kwargs["query_labels"][inliers])),
        )
        acc_values = []
        auprs = []
        losses = []
        support_losses = []
        query_losses = []
        inlier_entropies = []
        soft_assignement_entropies = []
        inlier_scores_means = []
        inlier_scores_stds = []

        N = query_features.size(0) + support_features.size(0)
        _edge_index = []
        for i in range(N):
            for j in range (N):
                _edge_index.append([i, j])

        edge_index = torch.tensor(_edge_index, dtype=torch.long)
        all_features = torch.cat([support_features, query_features], 0)
        in_dim = all_features.size(1)

        p_model = GAT(in_dim, num_classes)
        optimizer = torch.optim.Adam(p_model.parameters(), lr=0.005, weight_decay=5e-4)


        for _ in range(self.inference_steps):
            # Compute inlier scores
            # logits_q = self.get_logits(
            #     prototypes, query_features
            # )  # [query_size, num_classes]

            logits_q = p_model.get_log_prob(all_features, edge_index)[support_size:]


            inlier_scores = (
                self.ema_weight
                * (
                    (soft_assignements * logits_q / self.lambda_s)
                    .sum(-1, keepdim=True)
                    .sigmoid()
                )
                + (1 - self.ema_weight) * inlier_scores
            )  # [query_size, 1]

            # Compute new assignements
            soft_assignements = (
                (
                    self.ema_weight
                    * ((inlier_scores * logits_q / self.lambda_z).softmax(-1))
                    + (1 - self.ema_weight) * soft_assignements
                )
                if self.use_inlier_latent
                else (
                    self.ema_weight * ((logits_q / self.lambda_z).softmax(-1))
                    + (1 - self.ema_weight) * soft_assignements
                )
            )  # [query_size, num_classes]

            # Compute metrics
            outlier_scores = 1 - inlier_scores
            acc = (
                (soft_assignements.argmax(-1) == kwargs["query_labels"])[inliers]
                .float()
                .mean()
                .item()
            )
            acc_values.append(acc)
            precision, recall, thresholds = precision_recall_curve(
                outliers.numpy(), outlier_scores.numpy()
            )
            aupr = auc_fn(recall, precision)
            auprs.append(aupr)
            precision, recall, thresholds = precision_recall_curve(
                (~inliers).numpy(), outlier_scores.numpy()
            )

            all_log_prob = p_model.get_log_prob(all_features, edge_index)
            support_loss = soft_cross_entropy(
                all_log_prob[:support_size], one_hot_labels
            )
            query_loss = soft_cross_entropy(
                all_log_prob[support_size:],
                soft_assignements,
                inlier_scores,
            )
            inlier_entropy = binary_entropy(inlier_scores)
            soft_assignement_entropy = entropy(soft_assignements)
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            inlier_entropies.append(inlier_entropy)
            soft_assignement_entropies.append(soft_assignement_entropy)
            losses.append(
                support_loss
                + query_loss
                + self.lambda_s * inlier_entropy
                + self.lambda_z * soft_assignement_entropy
            )
            inlier_scores_means.append(inlier_scores.mean())
            inlier_scores_stds.append(inlier_scores.std())

            # Compute new A
            loss = support_loss + query_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        kwargs["intra_task_metrics"]["main_metrics"]["acc"].append(acc_values)
        kwargs["intra_task_metrics"]["main_metrics"]["aupr"].append(auprs)
        kwargs["intra_task_metrics"]["secondary_metrics"]["losses"].append(losses)
        kwargs["intra_task_metrics"]["secondary_metrics"]["support_losses"].append(
            support_losses
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["query_losses"].append(
            query_losses
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_entropies"].append(
            inlier_entropies
        )
        kwargs["intra_task_metrics"]["secondary_metrics"][
            "soft_assignement_entropies"
        ].append(soft_assignement_entropies)
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_scores_means"].append(
            inlier_scores_means
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_scores_stds"].append(
            inlier_scores_stds
        )

        all_log_prob = p_model.get_log_prob(all_features, edge_index)
        logits_s = all_log_prob[:support_size]
        logits_q = all_log_prob[support_size:]

        if self.inference_steps == 0:
            outlier_scores = (
                1
                - (soft_assignements * logits_q / self.lambda_s)
                .sum(-1, keepdim=True)
                .sigmoid()
            )

        return (
            logits_s.softmax(-1),
            logits_q.softmax(-1),
            outlier_scores,
        )


def soft_cross_entropy(logits, soft_labels, _inlier_scores=None):
    _inlier_scores = (
        _inlier_scores if _inlier_scores is not None else torch.ones(len(logits))
    )
    return -((logits * soft_labels).sum(dim=1) * _inlier_scores).mean()


def binary_entropy(scores):
    return -(scores * scores.log() + (1 - scores) * (1 - scores).log()).mean()


def entropy(scores):
    return -(scores * scores.log()).sum(dim=1).mean()
