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
        self.digits = self.generate_digits(num_samples)

    # def __call__(self, x):
    #     if cuda:
    #         x = x.cuda()
    #     with torch.no_grad():
    #         x_hat, mu, logvar = self.model(x)
    #     return x_hat, mu, logvar
        
    def generate_digits(self, num_samples):
        digits = defaultdict(torch.Tensor)
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                # if cuda:
                #     batch_x = batch_x.cuda()
                x_hat, mu, sigma = self.model(batch_x)
                for x, y in zip(x_hat, batch_y):
                    if len(digits[y.item()]) < num_samples:
                        # add the generated image to the list
                        digits[y.item()] = torch.cat((digits[y.item()], x.unsqueeze(0)), dim=0)
                    else:
                        continue
                    
        return digits