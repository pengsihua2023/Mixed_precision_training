# Mixed_precision_training
### 原始代码  
```
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os

class MyModel(LightningModule):
    def __init__(self, model, train_dataset, test_dataset, batch_size=1, lr=1e-5, accumulation_steps=10):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, except 'model'
        self.automatic_optimization = False  # 禁用自动优化
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.accumulation_steps = accumulation_steps  # New attribute for gradient accumulation steps

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Gradient accumulation
        loss = loss / self.accumulation_steps  # Scale loss
        self.manual_backward(loss)  # Backward pass (accumulates gradients)
        
        # Condition to check if it is time to update weights
        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader()):
            self.optimizer.step()  # Update weights
            self.optimizer.zero_grad()  # Clear gradients
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)  # Save optimizer as attribute
        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8)

# Assuming the model and dataset are already loaded and prepared as per your requirements
model = GenSLM("genslm_2.5B_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sequences = [str(record.seq) for record in SeqIO.parse("./data/H3N2_upto_2023_01_23.nucleotide.fasta", "fasta")]
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)

# Split the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize LightningModule with the consistent batch_size and accumulation_steps
my_model = MyModel(model, train_dataset, test_dataset, batch_size=1, accumulation_steps=10)

# Define a checkpoint callback to save the best model
checkpoint_dir = './checkpoints/InflunzaA_gradient'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=checkpoint_dir,
    filename='InfluenzaA_2.5B_model_Gradient_accum-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
)

# Initialize the Trainer
trainer = Trainer(
    gpus=-1 if torch.cuda.is_available() else None,
    strategy='ddp',
    max_epochs=3,
    callbacks=[checkpoint_callback],
    default_root_dir=checkpoint_dir,    
)

# Start training
trainer.fit(my_model)

# Load the best model checkpoint
best_model_path = checkpoint_callback.best_model_path
best_model = MyModel.load_from_checkpoint(best_model_path)

# Save the best model for inference
torch.save(best_model.model.state_dict(), os.path.join(checkpoint_dir, "Model_2.5B_gradient-accum.pth"))


```
实现混合精度训练非常简单，特别是使用了 PyTorch Lightning 的情况下。PyTorch Lightning 提供了非常简便的方式来启用混合精度训练，几乎不需要对现有代码进行修改。混合精度训练通过利用浮点16位（FP16）进行计算来减少内存使用并加速训练，同时保持关键部分的计算在更高精度（如浮点32位，FP32）上以维持训练稳定性和模型精度。  

要在 PyTorch Lightning 中启用混合精度训练，您只需要在创建 Trainer 实例时设置 precision 参数为 16。这会告诉 Lightning 使用 FP16 训练，而不是默认的 FP32。同时，为了最大化性能和兼容性，建议启用 NVIDIA 的 Apex 库（如果可用）。  

以下是修改后的代码段，展示了如何在 Trainer 初始化时启用混合精度  

### 修改后的代码
```
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os

class MyModel(LightningModule):
    def __init__(self, model, train_dataset, test_dataset, batch_size=1, lr=1e-5, accumulation_steps=10):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, except 'model'
        self.automatic_optimization = False  # 禁用自动优化
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.accumulation_steps = accumulation_steps  # New attribute for gradient accumulation steps

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Gradient accumulation
        loss = loss / self.accumulation_steps  # Scale loss
        self.manual_backward(loss)  # Backward pass (accumulates gradients)
        
        # Condition to check if it is time to update weights
        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader()):
            self.optimizer.step()  # Update weights
            self.optimizer.zero_grad()  # Clear gradients
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)  # Save optimizer as attribute
        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8)

# Assuming the model and dataset are already loaded and prepared as per your requirements
model = GenSLM("genslm_2.5B_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sequences = [str(record.seq) for record in SeqIO.parse("./data/H3N2_upto_2023_01_23.nucleotide.fasta", "fasta")]
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)

# Split the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize LightningModule with the consistent batch_size and accumulation_steps
my_model = MyModel(model, train_dataset, test_dataset, batch_size=1, accumulation_steps=10)

# Define a checkpoint callback to save the best model
checkpoint_dir = './checkpoints/InflunzaA_gradient'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=checkpoint_dir,
    filename='InfluenzaA_2.5B_model_Gradient_accum-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
)

# Initialize the Trainer
trainer = Trainer(
    gpus=-1 if torch.cuda.is_available() else None,
    strategy='ddp',
    max_epochs=3,
    callbacks=[checkpoint_callback],
    default_root_dir=checkpoint_dir,
    precision=16,
)

# Start training
trainer.fit(my_model)

# Load the best model checkpoint
best_model_path = checkpoint_callback.best_model_path
best_model = MyModel.load_from_checkpoint(best_model_path)

# Save the best model for inference
torch.save(best_model.model.state_dict(), os.path.join(checkpoint_dir, "Model_2.5B_gradient-accum.pth"))
```
通过这个简单的修改，PyTorch Lightning 会自动处理大多数与混合精度相关的细节，包括自动转换数据类型和管理梯度缩放等，这有助于防止在使用 FP16 时可能出现的梯度下溢问题。  

请注意，混合精度训练主要由NVIDIA的GPU支持，如果您使用的是支持Tensor Cores的GPU（如Volta、Turing、Ampere架构），那么从混合精度训练中获得的性能提升会更加显著。  

最后，虽然混合精度训练可以显著减少内存使用并加速训练，但在某些情况下可能需要对训练过程进行微调，以确保模型收敛性和最终精度不受影响。这可能包括调整学习率、使用更精确的梯度累积策略等。  
