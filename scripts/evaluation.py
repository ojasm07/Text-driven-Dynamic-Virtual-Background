import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.models import inception_v3
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
from clip import CLIP
from PIL import Image


class InceptionScore:
    def __init__(self, model=None):
        if model is None:
            self.model = inception_v3(pretrained=True, transform_input=False)
        else:
            self.model = model

        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def preprocess_images(self, images):
        preprocessed_images = torch.stack(
            [self.transform(img) for img in images])
        return preprocessed_images

    def calculate_inception_score(self, generated_images, batch_size=32, splits=10):
        generated_images = self.preprocess_images(generated_images)

        with torch.no_grad():
            preds = self.model(generated_images).squeeze()

        scores = []
        for i in tqdm(range(splits)):
            start_idx = i * (len(preds) // splits)
            end_idx = (i + 1) * (len(preds) //
                                 splits) if i != splits - 1 else len(preds)
            scores.append(
                entropy(F.softmax(preds[start_idx:end_idx], dim=1).mean(dim=0)))

        is_score = np.exp(np.mean(scores))

        return is_score


class FIDScoreCalculator:
    def __init__(self, real_data_loader, fake_data_loader, device="cuda"):
        self.device = device
        self.real_data_loader = real_data_loader
        self.fake_data_loader = fake_data_loader

        self.inception_model = inception_v3(
            pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        self.inception_model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def preprocess_images(self, images):
        preprocessed_images = torch.stack(
            [self.transform(img) for img in images])
        return preprocessed_images.to(self.device)

    def calculate_fid(self):
        real_features, fake_features = [], []

        with torch.no_grad():
            for real_batch in self.real_data_loader:
                real_batch = self.preprocess_images(real_batch)
                features = self.inception_model(real_batch)
                real_features.append(features)

        real_features = torch.cat(real_features, dim=0)

        with torch.no_grad():
            for fake_batch in self.fake_data_loader:
                fake_batch = self.preprocess_images(fake_batch)
                features = self.inception_model(fake_batch)
                fake_features.append(features)

        fake_features = torch.cat(fake_features, dim=0)

        fid_score = self.calculate_frechet_distance(
            real_features, fake_features)

        return fid_score.item()

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        eps = 1e-6
        mu1, mu2 = mu1.cpu().numpy(), mu2.cpu().numpy()
        sigma1, sigma2 = sigma1.cpu().numpy(), sigma2.cpu().numpy()

        diff = mu1 - mu2

        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_score = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid_score


class CLIPScoreCalculator:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.clip_model, self.clip_tokenizer = CLIP(model_name).to(device)
        self.clip_model.eval()

        self.image_transform = Compose([
            Resize((224, 224)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def calculate_clip_score(self, image_path, text_description):
        image_tensor = self.preprocess_image(image_path)
        text_input = self.clip_tokenizer(
            [text_description], return_tensors="pt").to(self.device)

        image_features = self.clip_model.encode_image(image_tensor)
        text_features = self.clip_model.encode_text(**text_input)

        similarity_score = (
            100.0 * torch.nn.functional.cosine_similarity(image_features, text_features)).item()

        return similarity_score
