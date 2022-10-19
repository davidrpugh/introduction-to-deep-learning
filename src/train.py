import os
import pathlib

from sklearn import metrics
import torch
from torch import nn, optim, utils
from torchvision import datasets, models, transforms
from tqdm import tqdm


BATCH_SIZE = 256
DATA_DIR = pathlib.Path("data/")
DATALOADER_NUM_WORKERS = 6
DEVICE = torch.device("cuda")
NUM_CLASSES = 10
NUM_TRAIN_EPOCHS = 10
OPTIMIZER_LEARNING_RATE = 1e-3
OPTIMIZER_MOMENTUM = 0.9
OUTPUT_DIR = pathlib.Path("results/example-training-job/")
OUTPUT_FILENAME = OUTPUT_DIR / "model.pt"
PREFETCH_FACTOR = 2
RESIZE_SIZE = 224
SEED = 42
TQDM_DISABLE = True


# create the output directory
if not OUTPUT_DIR.exists():
    os.mkdir(OUTPUT_DIR)

# set seed for reproducibility
torch.manual_seed(SEED)

# create the train and test datasets
_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root=DATA_DIR,
                                 train=True,
                                 download=True,
                                 transform=_transform)

test_dataset = datasets.CIFAR10(root=DATA_DIR,
                                train=False,
                                download=True,
                                transform=_transform)

# create the train and test dataloaders
train_dataloader = (utils.data
                         .DataLoader(train_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=DATALOADER_NUM_WORKERS,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     prefetch_factor=PREFETCH_FACTOR))
test_dataloader = (utils.data
                        .DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=DATALOADER_NUM_WORKERS,
                                    persistent_workers=True,
                                    pin_memory=True,
                                    prefetch_factor=PREFETCH_FACTOR))

# define a model_fn, loss function, and an optimizer
model_fn = models.resnet50(pretrained=False,
                           num_classes=NUM_CLASSES)
model_fn.to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_fn.parameters(),
                      lr=OPTIMIZER_LEARNING_RATE,
                      momentum=OPTIMIZER_MOMENTUM)

# train the model
print("Training started...")
for epoch in range(NUM_TRAIN_EPOCHS):

    with tqdm(train_dataloader, unit="batch", disable=TQDM_DISABLE) as tepoch:

        for (features, targets) in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predictions = model_fn(features.to(DEVICE))
            loss = loss_fn(predictions, targets.to(DEVICE))
            loss.backward()
            optimizer.step()

print("...training finished!")

# save the trained model
torch.save(model_fn.state_dict(), OUTPUT_FILENAME)

# compute the predications on the test data
batch_targets = []
batch_predicted_targets = []

with torch.no_grad():
    for (features, targets) in test_dataloader:
        predicted_probs = model_fn(features.to(DEVICE))
        predicted_targets = predicted_probs.argmax(axis=1)
        batch_targets.append(targets)
        batch_predicted_targets.append(predicted_targets)

# generate a classification report
test_target = (torch.cat(batch_targets)
                    .cpu())
test_predicted_targets = (torch.cat(batch_predicted_targets)
                               .cpu())

classification_report = metrics.classification_report(
    test_target,
    test_predicted_targets,
)
print(classification_report)
