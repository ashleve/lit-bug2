
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log("train/loss", loss, on_epoch=True)
        return {"loss": loss, "wowoowowo": "wowo"}

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss, "wowoowowo": "wowo"}

    # uncomment this to fix the bug
    # def test_epoch_end(self, outputs) -> None:
    #     torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


num_samples = 10000

train = RandomDataset(32, num_samples)
train = DataLoader(train, batch_size=32, num_workers=12)

val = RandomDataset(32, num_samples)
val = DataLoader(val, batch_size=32, num_workers=12)

test = RandomDataset(32, num_samples)
test = DataLoader(test, batch_size=32, num_workers=12)

model = BoringModel()

ckpt = pl.callbacks.ModelCheckpoint(monitor="haha")

trainer = pl.Trainer(
    min_epochs=1, 
    max_epochs=1, 
)

trainer.fit(model, train, val)

print(trainer.callback_metrics)

trainer.test(test_dataloaders=test)
