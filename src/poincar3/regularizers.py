import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distrib import is_distributed


class KoLeoLoss(torch.nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. -
    2018 - Spreading vectors for similarity search. Copied verbatim from
    DINOv2's `dinov2/loss/koleo_loss.py` (see `loss.py`'s
    `Cfg.koleo_loss_weight` for how it's wired in -- applied to the raw,
    pre-`global_head` camera-token scene embedding, the closest Poincar3 analogue
    of DINOv2's raw backbone CLS token that `ssl_meta_arch.py` applies this to).
    """

    def __init__(self):
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


@torch.no_grad()
def sinkhorn_knopp_teacher(
    teacher_logits: torch.Tensor, teacher_temp: float, sinkhorn_iterations: int = 3
) -> torch.Tensor:
    """SwAV/iBOT-style Sinkhorn-Knopp optimal transport, matching
    DINOv2's `dinov2/loss/ibot_patch_loss.py`'s `sinkhorn_knopp_teacher`.

    `teacher_logits`: `[num_samples, num_prototypes]`. Q is a direct function of
    exactly the samples given here, no running statistic across steps. Shared
    by `Poincar3Loss` and `poincar3.baselines.dino_objective_loss.DinoObjectiveLoss`
    so both use the identical collapse-prevention mechanism.
    """
    Q = torch.exp(teacher_logits.float() / teacher_temp).t()  # [D, num_samples]
    num_prototypes, num_samples = Q.shape

    num_samples = torch.as_tensor(num_samples, device=Q.device, dtype=Q.dtype)
    if is_distributed():
        dist.all_reduce(num_samples)

    sum_Q = Q.sum()
    if is_distributed():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        sum_of_rows = Q.sum(dim=1, keepdim=True)
        if is_distributed():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= num_prototypes

        Q /= Q.sum(dim=0, keepdim=True)
        Q /= num_samples

    Q *= num_samples
    return Q.t()
