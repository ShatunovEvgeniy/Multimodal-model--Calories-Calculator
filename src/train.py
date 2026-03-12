from tqdm import tqdm
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from torch.optim import AdamW, lr_scheduler
import torch
import torch.nn as nn

from utils import seed_everything, set_requires_grad
from dataset import prepare_dataloaders
from model import CaloriesModel


def train(config, device, weights_path=None):
    seed_everything(config.SEED)

    # Инициализация модели
    model = CaloriesModel(config).to(device)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=False)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=False)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.head.parameters(),
        'lr': config.MLP_LR
    }])

    # Функция потерь и scheduler
    criterion = nn.SmoothL1Loss(beta=1.0)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Мы хотим минимизировать MAE
        factor=0.5,          # Во сколько раз уменьшать LR
        patience=5,          # Сколько эпох ждать улучшения перед уменьшением
    )

    # Загрузка данных
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(config)

    # Инициализация метрик
    mae_metric_train = MeanAbsoluteError().to(device)
    mape_metric_train = MeanAbsolutePercentageError().to(device)
    mae_metric_val = MeanAbsoluteError().to(device)
    mape_metric_val = MeanAbsolutePercentageError().to(device)

    best_mae_val = float('inf')

    print("Training started")

    # Обучение
    for epoch in tqdm(range(config.EPOCHS), desc="Epochs", leave=True):
        model.train()
        total_loss = 0.0

        # Прогресс-бар по батчам обучения
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch in train_pbar:
            inputs = {
                'ingrs': batch['ingrs'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            target = batch['target'].to(device)

            optimizer.zero_grad()
            prediction = model(**inputs)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _ = mae_metric_train(preds=prediction, target=target)
            _ = mape_metric_train(preds=prediction, target=target)

            # 🔹 Обновление описания бара в реальном времени
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (train_pbar.n + 1):.4f}"
            })

        # Вычисление метрик после эпохи
        train_mae = mae_metric_train.compute().cpu().numpy()
        train_mape = mape_metric_train.compute().cpu().numpy()
        val_mae, val_mape = validate(model, val_dataloader, device, mae_metric_val, mape_metric_val)

        # Обновление шага
        scheduler.step(val_mae)

        # Сброс метрик
        mae_metric_train.reset()
        mape_metric_train.reset()
        mae_metric_val.reset()
        mape_metric_val.reset()

        # Вывод итогов эпохи
        epoch_msg = (f"Epoch {epoch + 1}/{config.EPOCHS} | "
                     f"Loss: {total_loss / len(train_dataloader):.4f} | "
                     f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                     f"Train MAPE: {train_mape:.4f} | Val MAPE: {val_mape:.4f}")
        tqdm.write(epoch_msg)

        # Сохранение лучшей модели
        if val_mae < best_mae_val:
            best_mae_val = val_mae
            tqdm.write(f"✨ New best model at epoch {epoch + 1} (Val MAE: {val_mae:.4f})")
            torch.save(model.state_dict(), config.WEIGHTS_DIR / f"epoch_{epoch + 1}.pth")


def validate(model, val_loader, device, mae_metric, mape_metric):
    model.eval()

    # Прогресс-бар для валидации
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Val]", leave=False):
            inputs = {
                'ingrs': batch['ingrs'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            target = batch['target'].to(device)

            prediction = model(**inputs)
            _ = mae_metric(preds=prediction, target=target)
            _ = mape_metric(preds=prediction, target=target)

    return mae_metric.compute().cpu().numpy(), mape_metric.compute().cpu().numpy()


if __name__ == '__main__':
    from pathlib import Path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        SEED = 42

        # Параметры массы для нормализации (взяты из анализа)
        MASS_MEAN = 214.980
        MASS_STD = 161.497
        MASS_USE_LOG = False  # (max-min)=3050 при mean=215 - необходимо сделать распределение более симметричным

        # Параметры массы для нормализации (взяты из анализа)
        CAL_MEAN = 255.013
        CAL_STD = 219.638
        CAL_USE_LOG = False  # (max-min)=3943 при mean=255 - необходимо сделать распределение более симметричным

        # Модели
        TEXT_MODEL_NAME = "bert-base-uncased"
        IMAGE_MODEL_NAME = "resnet50"

        # Какие слои размораживаем - совпадают с неймингом в моделях
        TEXT_MODEL_UNFREEZE = "encoder.layer.9|encoder.layer.10|encoder.layer.11|pooler"
        IMAGE_MODEL_UNFREEZE = "layer4.|conv_head|bn2"

        # Гиперпараметры
        BATCH_SIZE = 32
        TEXT_LR = 3e-5  # LR для текстовой модели
        IMAGE_LR = 1e-4  # LR для изображений
        MLP_LR = 5e-4  # LR для классификатора
        EPOCHS = 100
        DROPOUT = 0.15
        HIDDEN_DIM = 256  # размерность проекции признаков моделей

        # Пути
        BASE_DIR = Path.cwd().parent
        DATASET_DIR = BASE_DIR / "data"
        IMAGES_DIR = DATASET_DIR / "images"
        DISH_CSV_PATH = DATASET_DIR / "dish.csv"
        INGREDIENTS_CSV_PATH = DATASET_DIR / "ingredients.csv"
        WEIGHTS_DIR = BASE_DIR / "models" / "calories_normalized_no_log"

    config = Config()
    train(config, device)
