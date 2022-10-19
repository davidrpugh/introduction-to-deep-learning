import argparse
import pathlib

from sklearn import metrics
import torch
from torch import nn, optim, utils
from torchvision import datasets, models, transforms
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size",
                    default=256,
                    type=int,
                    help="Number of training samples per batch.")
parser.add_argument("--checkpoint-filepath",
                    type=str,
                    help="Path to a file containing the current checkpoint")
parser.add_argument("--data-dir",
                    required=True,
                    type=str,
                    help="Path to directory containing the train, val, test data.")
parser.add_argument("--dataloader-num-workers",
                    required=True,
                    type=int,
                    help="Number of workers to use for loading data.")
parser.add_argument("--dataloader-prefetch-factor",
                    default=2,
                    type=int,
                    help="Number of data batches to prefetch per worker.")
parser.add_argument("--disable-gpu",
                    action="store_true",
                    help="Disable GPU(s) for training and inference.")
parser.add_argument("--num-training-epochs",
                    default=1,
                    type=int,
                    help="Number of training epochs.")
parser.add_argument("--optimizer-learning-rate",
                    default=1e-3,
                    type=float,
                    help="Learning rate for optimizer.")
parser.add_argument("--optimizer-momentum",
                    default=0.9,
                    type=float,
                    help="Momentum for optimizer.")
parser.add_argument("--seed",
                    type=int,
                    help="Seed used for pseudorandom number generation.")
parser.add_argument("--tqdm-disable",
                    action="store_true",
                    help="Disables the training progress bar.")
parser.add_argument("--write-checkpoint-to",
                    type=str,
                    help="Path to the file where checkpoint should be written")
args = parser.parse_args()


# no need to expose these as command line args
DATA_DIR = pathlib.Path(args.data_dir)
DEVICE = torch.device("cpu") if args.disable_gpu else torch.device("cuda")
NUM_CLASSES = 10
RESIZE_SIZE = 224


# set up checkpointing
if args.checkpoint_filepath is not None:
    CHECKPOINT_FILEPATH = pathlib.Path(args.checkpoint_filepath)
else:
    CHECKPOINT_FILEPATH = None

if args.write_checkpoint_to is not None:
    WRITE_CHECKPOINT_TO = pathlib.Path(args.write_checkpoint_to)
else:
    WRITE_CHECKPOINT_TO = None

# set seed for reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)

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
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.dataloader_num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     prefetch_factor=args.dataloader_prefetch_factor))
test_dataloader = (utils.data
                        .DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.dataloader_num_workers,
                                    persistent_workers=True,
                                    pin_memory=True,
                                    prefetch_factor=args.dataloader_prefetch_factor))

# define a model_fn, loss function, and an optimizer
model_fn = models.resnet50(pretrained=False,
                           num_classes=NUM_CLASSES)
model_fn.to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_fn.parameters(),
                      lr=args.optimizer_learning_rate,
                      momentum=args.optimizer_momentum)

# load model checkpoint (if available)
if CHECKPOINT_FILEPATH is not None:
    checkpoint_file = torch.load(CHECKPOINT_FILEPATH)
    model_fn.load_state_dict(checkpoint_file["model_state_dict"])
    optimizer.load_state_dict(checkpoint_file["optimizer_state_dict"])

# train the model
print("Training started...")
for epoch in range(args.num_training_epochs):

    with tqdm(train_dataloader, unit="batch", disable=args.tqdm_disable) as tepoch:

        for (features, targets) in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predictions = model_fn(features.to(DEVICE))
            loss = loss_fn(predictions, targets.to(DEVICE))
            loss.backward()
            optimizer.step()
    
    if WRITE_CHECKPOINT_TO is not None:
        checkpoint = {
            "model_state_dict": model_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, WRITE_CHECKPOINT_TO)

print("...training finished!")

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
