import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    save_checkpoint,
    load_checkpoint,
    plot_to_tensorboard,
    gradient_penalty
)

from model.Generator import Generator
from model.Discriminator import Discriminator
import config
from math import log2
from tqdm import tqdm

torch.backends.cudnn.benchmarks = True


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNEL_IMG)],
                [0.5 for _ in range(config.CHANNEL_IMG)],
            ),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return loader, dataset


def train(gen, critic, loader, dataset, step, alpha, opt_gen, opt_critic, tensorboard_step, writer, scaler_gen,
          scaler_critic):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)

        cur_batch_size = real.shape[0]

        # Train Discriminator, maximize E[critic(real)] - E[critic(fake)]
        noise = torch.rand(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, device=config.DEVICE)
            loss_critic = (
                - torch.mean(critic_real) - torch.mean(critic_fake)
                + config.LAMBDA_GP* gp
                + (0.0001 * torch.mean(critic_real) ** 2)
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator, maximize E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = - torch.mean(gen_fake)

        opt_critic.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_critic)
        scaler_gen.update()

        alpha += cur_batch_size / len(dataset) * config.PROGRESSIVE_EPOCHS[step] * 0.5
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5

            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step
            )

            tensorboard_step += 1

    return tensorboard_step, alpha


def main():
    gen = Generator(config.Z_DIM, config.IN_CHANNEL, config.CHANNEL_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNEL, img_channels=config.CHANNEL_IMG).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir="logs/gan")

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DIS, critic, opt_critic, config.LEARNING_RATE)

    gen.train(), critic.train()
    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_IMG_SIZE / 4))
    for num_epochs in range(config.PROGRESSIVE_EPOCHS[step:]):
        alpha = 1e-5
        loader, dataset = get_loader(4 * 2 ** step)
        print(f"Image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1} / {num_epochs}]")
            tensorboard_step, alpha = train(gen, critic, loader, dataset, step, alpha, opt_gen, opt_critic,
                                            tensorboard_step, writer, scaler_gen, scaler_critic)

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_DIS)

        step += 1


if __name__ == "__main__":
    main()


