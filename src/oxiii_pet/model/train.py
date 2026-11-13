import os
import time
from argparse import ArgumentParser
from pathlib import Path

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose, Resize, RandomResizedCrop, RandomHorizontalFlip,
    ToImage, ToDtype, Normalize,
)
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

from oxiii_pet.model.models import Resnet50
from oxiii_pet.dataset.dataset_oxiiit_pet import DEFAULT_SPLIT_FILE, OxfordData
from oxiii_pet.utils.run_utils import (
    MetricsCSV, default_run_dir, final_test_evaluation, get_class_names, get_rng_state, plot_confusion_matrix,
    run_epoch, set_rng_state, set_seed, setup_run,
)


def get_args():
    parser = ArgumentParser(description="CNN IN Oxiii Pet")
    parser.add_argument("--epochs","-e", type=int, default=10, help="number of epochs")
    parser.add_argument("--images_size","-i", type=int, default=224, help="size of images")
    parser.add_argument("--batch_size","-b", type=int, default=8, help="batch size")
    parser.add_argument("--dir","-d", type=str, default=os.getenv("DATA_DIR", "data"), help="data path")
    parser.add_argument("--run_dir","-r", type=str, default=None,
                        help="run folder for train.log, metrics.csv, config.json (default runs/<date>_scratch_<n>cls)")
    parser.add_argument("--logging","-l", type=str, default=None, help="TensorBoard dir (default <run_dir>/tensorboard)")
    parser.add_argument("--trained_model","-t", type=str, default=None, help="checkpoint dir (default <run_dir>/checkpoints)")
    parser.add_argument("--checkpoint","-c", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument("--num_class","-n", type=int, default=37)
    parser.add_argument("--clip","-cl", type=float, default=0.0, help="gradient-norm clipping; 0 disables")
    parser.add_argument("--seed","-s", type=int, default=42)
    parser.add_argument("--num_workers","-w", type=int, default=4)
    parser.add_argument("--split_file", type=str, default=DEFAULT_SPLIT_FILE,
                        help="JSON with the train/val image names of trainval.txt")
    args = parser.parse_args()
    return args


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

def main():
    args = get_args()
    args.run_dir = args.run_dir or default_run_dir(f"scratch_{args.num_class}cls")
    args.logging = args.logging or os.path.join(args.run_dir, "tensorboard")
    args.trained_model = args.trained_model or os.path.join(args.run_dir, "checkpoints")
    Path(args.trained_model).mkdir(parents=True, exist_ok=True)
    logger, _ = setup_run(args.run_dir, args, "scratch")
    generator, worker_init_fn = set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    num_epochs = args.epochs
    num_class = args.num_class
    DATA_DIR = args.dir
    image_size = args.images_size

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = Compose([
        RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        RandomHorizontalFlip(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize,
    ])
    test_transform = Compose([
        Resize((image_size, image_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize,
    ])
    train_data = OxfordData(num_class, DATA_DIR, split="train", transform=train_transform, split_file=args.split_file)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_data = OxfordData(num_class, DATA_DIR, split="val", transform=test_transform, split_file=args.split_file)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.batch_size,
                                num_workers=args.num_workers, drop_last=False, shuffle=False)
    # The test split is only used once, after the last epoch (final_test_evaluation).
    test_data = OxfordData(num_class, DATA_DIR, split="test", transform=test_transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=False, shuffle=False)
    class_names = get_class_names(num_class, DATA_DIR)
    logger.info("Train samples: %d | Val samples: %d | Test samples: %d (test used once, after training)",
                len(train_data), len(val_data), len(test_data))
    writer = SummaryWriter(args.logging)
    metrics = MetricsCSV(os.path.join(args.run_dir, "metrics.csv"))
    model = Resnet50(num_class).to(device)
    initialize_weights(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_epoch = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_val_accuracy = checkpoint["best_acc"]
        best_epoch = checkpoint.get("best_epoch")
        if checkpoint.get("rng_state") is not None:
            set_rng_state(checkpoint["rng_state"], generator)
            logger.info("Resumed from %s at epoch %d (RNG state restored)", args.checkpoint, start_epoch)
        else:
            # Older checkpoints have no RNG state. Without this reseed, set_seed(seed) above would make the
            # next epoch replay the shuffle order and augmentations of epoch 1.
            set_seed(args.seed * 1000 + start_epoch)
            generator.manual_seed(args.seed * 1000 + start_epoch)
            logger.warning("Resumed from %s at epoch %d; checkpoint has no RNG state, reseeded with %d "
                           "(data order differs from an uninterrupted run)",
                           args.checkpoint, start_epoch, args.seed * 1000 + start_epoch)
    else:
        start_epoch = 0
        best_val_accuracy = -1.0  # so epoch 1 is always saved as best, even at 0% accuracy
    num_iters = len(train_dataloader)
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, _, _ = run_epoch(
            model, train_dataloader, criterion, device, optimizer=optimizer,
            desc=f"Train {epoch + 1}/{num_epochs}", clip=args.clip,
            writer=writer, global_step=epoch * num_iters)
        scheduler.step()
        val_loss, val_accuracy, all_predictions, all_labels = run_epoch(
            model, val_dataloader, criterion, device, desc=f"Val   {epoch + 1}/{num_epochs}")
        plot_confusion_matrix(writer, confusion_matrix(all_labels.numpy(), all_predictions.numpy(),
                                                       labels=list(range(num_class))), class_names, epoch)

        is_best = val_accuracy > best_val_accuracy  # strict: ties keep the earlier epoch
        if is_best:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_val_accuracy,
            "best_epoch": best_epoch,
            "val_acc": val_accuracy,
            "num_class": num_class,
            "class_names": class_names,
            "rng_state": get_rng_state(generator),
        }
        torch.save(checkpoint, "{}/last_model.pth".format(args.trained_model))
        if is_best:
            torch.save(checkpoint, "{}/best_model.pth".format(args.trained_model))

        epoch_time = time.time() - epoch_start
        writer.add_scalar("Train/EpochLoss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
        writer.add_scalar("LR", lr, epoch)
        metrics.append(epoch=epoch + 1, lr=lr, train_loss=round(train_loss, 6), train_acc=round(train_acc, 6),
                       val_loss=round(val_loss, 6), val_acc=round(val_accuracy, 6),
                       epoch_time_s=round(epoch_time, 1), is_best=int(is_best))
        logger.info(
            "Epoch %d/%d | lr %.3e | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.1fs%s",
            epoch + 1, num_epochs, lr, train_loss, train_acc, val_loss, val_accuracy, epoch_time,
            " | new best val acc, saved best_model.pth" if is_best else "")
    writer.close()

    final_test_evaluation(
        model, test_dataloader, criterion, device, args.trained_model, args.run_dir, num_class,
        class_names, logger, extra={"n_train": len(train_data), "n_val": len(val_data),
                                    "split_file": args.split_file, "seed": args.seed})

if __name__ == '__main__':
    main()
