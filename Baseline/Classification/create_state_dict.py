import torch



model = torch.load("model2.pt")

new_model = torch.save(model.state_dict(), "model_state_dict_model2.pt")