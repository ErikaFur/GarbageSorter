import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def accuracy(outputs, labels):
    return torch.tensor(torch.sum(outputs.argmax(1) == labels.argmax(1)).item() / len(labels))

#implement Base for Classification
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        #print(batch.shape)
        #print()
        #batch - (img, label)
        self.train()
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        #batch - (img, label)
        self.eval()
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}


    def validation_epoch_end(self, outputs):
        #outputs - list of dicts of validations
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch+1}: train_loss: {result['train_loss']}, val_loss: {result['val_loss']}, val_acc: {result['val_acc']}")

class ImageClassificationInception(nn.Module):
    def training_step(self, batch):
        #print(batch.shape)
        #print()
        #batch - (img, label)
        self.train()
        images, labels = batch
        outputs = self(images)
        loss1 = F.cross_entropy(outputs, labels)
        loss = loss1
        return loss

    def validation_step(self, batch):
        #batch - (img, label)
        self.eval()
        images, labels = batch
        outputs = self(images)
        loss1 = F.cross_entropy(outputs, labels)
        loss = loss1
        acc = accuracy(outputs, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}


    def validation_epoch_end(self, outputs):
        #outputs - list of dicts of validations
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch+1}: train_loss: {result['train_loss']}, val_loss: {result['val_loss']}, val_acc: {result['val_acc']}")


classDecode = {1:"paper/cardboard", 2: "metal", 3: "plastic", 4: "glass"}
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classDecode.values()))

    def forward(self, xb):
        o = self.network(xb)
        return F.softmax(o)

class DenseNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.densenet121(pretrained=True)
        self.network.classifier = nn.Linear(1024, len(classDecode.values()))

    def forward(self, xb):
        o = self.network(xb)
        return F.softmax(o)

class InceptionNet(ImageClassificationInception):
    def __init__(self):
        super().__init__()
        self.network = models.inception_v3(pretrained=True)
        self.network.aux_logits=False
        num_ftrs = self.network.AuxLogits.fc.in_features
        self.network.AuxLogits.fc = nn.Linear(num_ftrs, len(classDecode.values()))
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs,len(classDecode.values()))

    def forward(self, xb):
        o = self.network(xb)
        return F.softmax(o)

class VGGNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.vgg16(pretrained=True)
        # Replace last layer
        self.network.classifier[6] = torch.nn.Linear(4096, len(classDecode.values()))

    def forward(self, xb):
        o = self.network(xb)
        return F.softmax(o)

class Net(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(16, 16, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 4 * 4 , 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return F.softmax(x)
