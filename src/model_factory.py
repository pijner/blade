import time
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using GPU
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TorchMLP(nn.Module):
    def __init__(self, num_cont_features, cat_dims, embed_dims, num_classes):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, embed_dims)]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embed_total_dim = sum(embed_dims)
        input_dim = num_cont_features + embed_total_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_cont, x_cat):
        if x_cat.dtype != torch.long:
            x_cat = x_cat.long()
        assert x_cat.ndim == 2, f"x_cat shape should be (batch_size, num_cat_features), got {x_cat.shape}"
        assert x_cont.ndim == 2, f"x_cont shape should be (batch_size, num_cont_features), got {x_cont.shape}"

        x_cont = x_cont.to(self.device)
        x_cat = x_cat.to(self.device)

        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        for i, emb_out in enumerate(x):
            assert emb_out.device == x_cont.device, f"Mismatch at emb {i}: {emb_out.device} vs {x_cont.device}"
            assert emb_out.shape[0] == x_cont.shape[0], f"Batch size mismatch at emb {i}"

        x = torch.cat(x + [x_cont], dim=1)
        return self.model(x)


class TorchMLPWrapper:
    def __init__(self, num_cont_features, cat_dims, embed_dims, num_classes):
        self.model = TorchMLP(num_cont_features, cat_dims, embed_dims, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X_cont, X_cat, y_train, batch_size=1024, epochs=50, lr=1e-4):
        self.model.train()

        X_cont_tensor = torch.tensor(X_cont.values, dtype=torch.float32)
        X_cat_tensor = torch.tensor(X_cat.values, dtype=torch.long)
        y_tensor = torch.tensor(y_train.values, dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            start_time = time.time()
            perm = torch.randperm(X_cont_tensor.size(0))
            total_loss = 0.0

            for i in tqdm(range(0, X_cont_tensor.size(0), batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                idx = perm[i : i + batch_size]
                batch_x_cont = X_cont_tensor[idx]
                batch_x_cat = X_cat_tensor[idx]
                batch_y = y_tensor[idx]

                optimizer.zero_grad()
                logits = self.model(batch_x_cont, batch_x_cat)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x_cont.size(0)

            avg_loss = total_loss / X_cont_tensor.size(0)
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_cont_tensor, X_cat_tensor).argmax(dim=1)
                acc = (y_pred == y_tensor).float().mean().item()
            duration = time.time() - start_time
            print(
                f"[{time.strftime('%H:%M:%S')}] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f} - Time: {duration:.1f}s"
            )
            self.model.train()

    def predict(self, X_cont, X_cat):
        self.model.eval()
        X_cont_tensor = torch.tensor(X_cont.values, dtype=torch.float32).to(self.device)
        X_cat_tensor = torch.tensor(X_cat.values, dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(X_cont_tensor, X_cat_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X_cont, X_cat):
        self.model.eval()
        X_cont_tensor = torch.tensor(X_cont.values, dtype=torch.float32).to(self.device)
        X_cat_tensor = torch.tensor(X_cat.values, dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(X_cont_tensor, X_cat_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()


class TabNetWrapper:
    def __init__(self, input_dim, num_classes):
        self.model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax",  # 'sparsemax' also possible
            verbose=1,
            device_name="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.input_dim = input_dim
        self.num_classes = num_classes

    def fit(self, X_train, y_train):
        self.model.fit(X_train.values, y_train.values, max_epochs=10, patience=20, batch_size=1024)

    def predict(self, X):
        return self.model.predict(X.values)

    def predict_proba(self, X):
        return self.model.predict_proba(X.values)


MODEL_FACTORY = {
    "mlp": lambda *args: TorchMLPWrapper(*args),
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
    "svm": SVC(probability=True, class_weight="balanced", random_state=42),
    "xgb": XGBClassifier(scale_pos_weight=1.0),
    "tabnet": lambda input_dim, num_classes: TabNetWrapper(input_dim, num_classes),
}
