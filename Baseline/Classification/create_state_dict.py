import torch



model = torch.load("model3.pt")

new_model = torch.save(model.state_dict(), "model_state_dict3.pt")