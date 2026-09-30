import matplotlib.pyplot as plt, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from poincar3 import Poincar3

paths = ["assets/0015_A.jpg", "assets/0015_B.jpg", "assets/0015_C.jpg"]
views = torch.stack([
    torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((448, 448)))).permute(2, 0, 1) / 255.0
    for p in paths
]).cuda()

# One forward pass over all three views, so cross-view attention is actually exercised.
model = Poincar3().eval().cuda()
with torch.no_grad():
    _, features, _, _ = model(views[None])           # [1, 3, 784, 1024]
features = F.normalize(features[0], dim=-1).reshape(3, 28, 28, -1)

query = features[0, 11, 11]                          # one patch of view 0
maps = [(features[i] @ query).cpu().numpy() for i in range(3)]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, view, corr in zip(axes, views.cpu(), maps):
    ax.imshow(view.permute(1, 2, 0))
    ax.imshow(np.kron(corr, np.ones((16, 16))), cmap="turbo", alpha=0.6)
    ax.axis("off")
axes[0].plot(11 * 16 + 8, 11 * 16 + 8, "w*", markersize=16)
fig.savefig("correlation.jpg", bbox_inches="tight", dpi=150)