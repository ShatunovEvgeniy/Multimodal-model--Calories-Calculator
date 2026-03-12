import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

from src.dataset import prepare_dataloaders, denormalize_calories


def evaluate(model, config, device):
    """
    Оценка модели на тестовом наборе данных.

    Args:
        model: CaloriesModel — обученная модель
        config: Configuration файл
        device: torch.device — устройство для вычислений

    Returns:
        tuple: (test_mae, test_mape) — значения метрик в виде numpy.float32
    """
    _, _, test_loader = prepare_dataloaders(config)
    # Инициализация метрик, если не переданы
    mae_metric = MeanAbsoluteError().to(device)
    mape_metric = MeanAbsolutePercentageError().to(device)
    model.eval()
    model = model.to(device)

    print(f"\nStarting evaluation on test set ({len(test_loader)} batches)...")

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[Test]", leave=True)
        for batch in test_pbar:
            inputs = {
                'ingrs': batch['ingrs'].to(device, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
                'image': batch['image'].to(device, non_blocking=True),
                'mass': batch['mass'].to(device, non_blocking=True)
            }
            target = batch['target'].to(device, non_blocking=True)
            target = denormalize_calories(model.cal_stats, target)

            # Forward pass
            prediction = model.infer(**inputs)

            # Обновление метрик
            _ = mae_metric(preds=prediction, target=target)
            _ = mape_metric(preds=prediction, target=target)

            # Обновление прогресс-бара с текущими значениями
            test_pbar.set_postfix({
                "MAE": f"{mae_metric.compute().item():.4f}",
                "MAPE": f"{mape_metric.compute().item():.4f}"
            })

    # Вычисление финальных значений метрик
    test_mae = mae_metric.compute().cpu().numpy()
    test_mape = mape_metric.compute().cpu().numpy()

    # Вывод результатов
    print(f"\n✅ Test Results:")
    print(f"   MAE:  {test_mae:.4f}")
    print(f"   MAPE: {test_mape:.4f}")

    # Сброс метрик для возможного повторного использования
    mae_metric.reset()
    mape_metric.reset()

    return test_mae, test_mape


@torch.no_grad()
def get_worst_predictions(model, config, device, top_k=5, display_images=True):
    """
    Находит блюда с наибольшими ошибками предсказания на тестовом наборе.
    Возвращает DataFrame с результатами и отображает изображения.

    Args:
        model: CaloriesModel — обученная модель
        config: Configuration — конфиг с путями и параметрами данных
        device: torch.device — устройство для вычислений
        top_k: int — количество худших предсказаний для возврата
        display_images: bool — отображать ли изображения блюд

    Returns:
        pd.DataFrame — таблица с данными о худших предсказаниях
            Columns: dish_name, mass_g, ingredients, calories_true,
                     calories_pred, abs_error, rel_error_pct
    """
    # Загружаем тестовый даталоадер
    _, _, test_loader = prepare_dataloaders(config)

    # Инициализация токенайзера для декодирования ингредиентов
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    model.eval()
    results = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="[Analyzing]", leave=True)):
        inputs = {
            'ingrs': batch['ingrs'].to(device, non_blocking=True),
            'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
            'image': batch['image'].to(device, non_blocking=True),
            'mass': batch['mass'].to(device, non_blocking=True).float(),
        }
        target = batch['target'].to(device, non_blocking=True)

        prediction = model(**inputs)
        abs_error = torch.abs(prediction - target)  # [B]
        rel_error = abs_error / (target + 1e-6) * 100  # MAPE в процентах

        # Сбор результатов по батчу
        batch_size = target.shape[0]
        for i in range(batch_size):
            # === Декодирование ингредиентов из токенов в текст ===
            # ingrs имеет форму [B, N, L] -> берём [i, :, :] -> [N, L]
            ingr_tokens = batch['ingrs'][i]  # [N, L]
            ingr_attn_mask = batch['attention_mask'][i]  # [N, L]

            ingredients_list = []
            for j in range(ingr_tokens.shape[0]):  # по каждому ингредиенту
                token_ids = ingr_tokens[j]  # [L]
                attn_mask = ingr_attn_mask[j]  # [L]

                # Фильтруем паддинг токены
                valid_tokens = token_ids[attn_mask.bool()]

                # Декодим в текст
                ingr_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()

                if ingr_text:  # добавляем только непустые ингредиенты
                    ingredients_list.append(ingr_text)

            ingredients_str = ", ".join(ingredients_list)

            # === Сохранение изображения для отображения ===
            image_tensor = batch['image'][i].cpu().clone()

            dish_info = {
                'mass_g': batch['mass'][i].item(),
                'ingredients': ingredients_str,
                'calories_true': target[i].item(),
                'calories_pred': prediction[i].item(),
                'abs_error': abs_error[i].item(),
                'rel_error_pct': rel_error[i].item(),
                'image_tensor': image_tensor,  # сохраняем для последующего отображения
            }
            results.append(dish_info)

    # Сортировка по абсолютной ошибке и возврат топ-k
    results.sort(key=lambda x: x['abs_error'], reverse=True)
    worst_k = results[:top_k]

    # Создание DataFrame (без image_tensor, т.к. это не сериализуется в CSV)
    df_worst = pd.DataFrame([
        {k: v for k, v in item.items() if k != 'image_tensor'}
        for item in worst_k
    ])

    # === Отображение изображений ===
    if display_images:
        n_images = len(worst_k)
        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
        if n_images == 1:
            axes = [axes]

        for idx, (ax, item) in enumerate(zip(axes, worst_k)):
            # Денормализация изображения
            img = item['image_tensor'].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)

            # Конвертация в PIL для отображения
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"#{idx + 1}\nΔ={item['abs_error']:.0f} kcal",
                         fontsize=9, pad=5)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return df_worst
