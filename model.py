def fit(self, parameters, config):
    self.set_parameters(parameters)
    optimizer = optim.SGD(self.model.parameters(), lr=0.01)
    train(self.model, self.train_loader, optimizer, epochs=2, noise_scale=0.05)

    # Save local model after training
    torch.save(self.model.state_dict(), f"client_{self.client_id}_model.pth")
    print(f"Client {self.client_id} model saved as client_{self.client_id}_model.pth")

    return self.get_parameters(config={}), len(self.train_loader.dataset), {}
