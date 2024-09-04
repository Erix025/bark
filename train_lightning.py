from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning as L
import timm
import pandas as pd
import torch


class DogClassifier(L.LightningModule):
    def __init__(self, model, lr=3e-4, weight_decay=5e-2, pretrained=True):
        super(DogClassifier, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head.weight.requires_grad = True
        self.model.head.bias.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return optimizer


torch.set_float32_matmul_precision("medium")
model = timm.create_model(
    "regnety_1280.swag_ft_in1k",
    pretrained=True,
    num_classes=120,
)
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
print(model)

dataset = datasets.ImageFolder(root="data/ImageFolder", transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, validate_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=32
)
validate_loader = DataLoader(
    validate_dataset, batch_size=batch_size, shuffle=False, num_workers=32
)

# Callbacks
checkpoint = ModelCheckpoint(
    monitor="val_acc", mode="max", save_top_k=1, filename="{epoch}-{val_acc:.2f}"
)
early_stopping = EarlyStopping(monitor="val_acc", patience=5, mode="max")

model = DogClassifier(model=model)
trainer = L.Trainer(max_epochs=30, callbacks=[checkpoint, early_stopping])

trainer.fit(
    model=model, train_dataloaders=train_loader, val_dataloaders=validate_loader
)

# Evaluate the model
# model.eval()
# transform = transforms.Compose(
#     [
#         transforms.Resize((448, 448)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# test_dataset = datasets.ImageFolder("data/test", transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=103)
# predictions = trainer.predict(model, test_loader)

# probs = []
# for batch_pred in predictions:
#     batch = batch_pred.softmax(dim=1).tolist()
#     for each in batch:
#         probs.append(each)

# # Save probs with filename of image
# files = [f for f, _ in test_dataset.imgs]
# for i, file in enumerate(files):
#     probs[i].insert(0, file.split("/")[-1].split(".")[0])

# columns = ["id"] + train_dataset.classes
# df = pd.DataFrame(probs, columns=columns)
# df.to_csv("submission.csv", index=False)
