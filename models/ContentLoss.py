from FeatureExtractor import *

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.feature_extractor = FeatureExtractor()

    def forward(self, sr, hr):
        # Enhanced pixel loss
        pixel_loss = 0.9 * self.l1_loss(sr, hr) + 0.1 * self.mse_loss(sr, hr)

        # reduced Multi-level perceptual loss
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        perceptual_loss = (
                0.5 * self.mse_loss(sr_features, hr_features) +
                0.5 * self.l1_loss(sr_features, hr_features)
        )

        return pixel_loss + 0.01 * perceptual_loss