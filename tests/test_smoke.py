import torch

from poincar3 import Poincar3, SSLModel
from poincar3.loss import Poincar3Loss


# A tiny architecture, so the test needs no network access and no GPU.
TINY = Poincar3.Cfg(
    embed_dim=64,
    depth=2,
    num_heads=2,
    num_register_tokens=2,
    register_attention_block_indices=[1],
    encoder_size="vits",
    ibot_head_out_dim=128,
    ibot_head_hidden_dim=64,
    ibot_head_bottleneck_dim=16,
    global_head_out_dim=128,
    global_head_hidden_dim=64,
    global_head_bottleneck_dim=16,
)


def test_public_api_imports() -> None:
    assert Poincar3 is not None and SSLModel is not None and Poincar3Loss is not None


def test_forward_shapes() -> None:
    model = Poincar3(TINY).eval()
    images = torch.rand(1, 3, 3, 64, 64)  # (batch, frames, 3, H, W)

    with torch.no_grad():
        patch_logits, patch_features, global_logits, camera_tokens = model(images)

    num_patches = (64 // model.patch_size) ** 2
    assert patch_logits.shape == (1, 3, num_patches, TINY.ibot_head_out_dim)
    assert patch_features.shape == (1, 3, num_patches, TINY.embed_dim)
    assert global_logits.shape == (1, 3, TINY.global_head_out_dim)
    assert camera_tokens.shape == (1, 3, TINY.embed_dim)
    assert torch.isfinite(patch_features).all()


def test_head_mask_gathers() -> None:
    model = Poincar3(TINY).eval()
    images = torch.rand(1, 2, 3, 64, 64)
    num_patches = (64 // model.patch_size) ** 2
    head_mask = torch.zeros(1, 2, num_patches, dtype=torch.bool)
    head_mask[0, 0, :5] = True

    with torch.no_grad():
        patch_logits, *_ = model(images, head_mask=head_mask)

    assert patch_logits.shape == (5, TINY.ibot_head_out_dim)


def test_block_outputs_and_camera_tokens() -> None:
    model = Poincar3(TINY).eval()
    images = torch.rand(1, 2, 3, 64, 64)

    with torch.no_grad():
        out = model(images, return_camera_tokens=True, capture_block_outputs=True)

    assert len(out) == 6
    assert len(out[5]) == 2 * TINY.depth


def test_attention_probe() -> None:
    model = Poincar3(TINY).eval()
    probe = model.attention_probe()  # last block is "register" here
    assert probe is None
    probe = model.attention_probe(block_index=0)
    assert probe is not None and probe.patch_token_start == TINY.num_register_tokens + 1


def test_ssl_step_is_differentiable() -> None:
    from poincar3.types import Batch

    model = SSLModel(TINY)
    loss_fn = Poincar3Loss(Poincar3Loss.Cfg())
    num_patches = (64 // model.student.patch_size) ** 2
    mask = torch.zeros(1, 2, num_patches, dtype=torch.bool)
    mask[0, :, :8] = True
    batch = Batch(
        imgs=torch.rand(1, 3, 3, 64, 64),  # teacher sees one extra frame
        student_imgs=torch.rand(1, 2, 3, 64, 64),
        mask=mask,
    )

    loss, stats = loss_fn(batch=batch, model=model, step=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert {"patch_loss", "global_loss", "koleo_loss"} <= stats.keys()
    assert any(p.grad is not None for p in model.student.parameters())


if __name__ == "__main__":
    test_public_api_imports()
    test_forward_shapes()
    test_head_mask_gathers()
    test_block_outputs_and_camera_tokens()
    test_attention_probe()
    test_ssl_step_is_differentiable()
    print("all smoke tests passed")
