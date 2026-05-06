class WeightedAverage(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        # Convert aggregated parameters back into a model
        if aggregated_parameters is not None:
            model = Net()
            state_dict = model.state_dict()
            for k, v in zip(state_dict.keys(), aggregated_parameters[0]):
                state_dict[k] = torch.tensor(v)
            model.load_state_dict(state_dict)

            # Save global model
            torch.save(model.state_dict(), f"global_model_round_{rnd}.pth")
            print(f"Global model saved as global_model_round_{rnd}.pth")

        return aggregated_parameters
