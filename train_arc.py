# train_arc.py
from autonomousai import AutonomousAI
import torch, json, numpy as np

model = AutonomousAI().cuda()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

# Dummy data (ganti dengan ARC-AGI)
data = json.load(open('arc_dummy.json'))

for epoch in range(100):
    total_loss = 0
    for item in data:
        obs = torch.randn(1, 768).cuda()
        action = torch.randn(1, 512).cuda()
        target = torch.randn(1, 768).cuda()

        out = model(obs, action)
        loss = F.mse_loss(out['pred'], target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(data):.4f}")
