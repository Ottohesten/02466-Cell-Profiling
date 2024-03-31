from collections import defaultdict
import torch

# to run inference on the trained model
class Inference:
    def __init__(self, model, test_loader, num_samples=10):
        self.model = model
        self.model.eval()
        self.model.to('cpu')
        self.num_samples = num_samples
        self.test_loader = test_loader
        self.samples = self.generate_samples(num_samples)

    # def __call__(self, x):
    #     if cuda:
    #         x = x.cuda()
    #     with torch.no_grad():
    #         x_hat, mu, logvar = self.model(x)
    #     return x_hat, mu, logvar
        
    def generate_samples(self, num_samples):
        samples = defaultdict(torch.Tensor)
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                # if cuda:
                #     batch_x = batch_x.cuda()
                x_hat, mu, sigma = self.model(batch_x)
                for x, y in zip(x_hat, batch_y):
                    if len(samples[str(y.item())]) < num_samples:
                        # add the generated image to the list
                        samples[str(y.item())] = torch.cat((samples[str(y.item())], x.unsqueeze(0)), dim=0)
                    else:
                        continue
                    
        return samples