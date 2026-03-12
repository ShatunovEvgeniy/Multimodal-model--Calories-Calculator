import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torch
import random

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial
from transformers import AutoTokenizer


def denormalize_calories(cal_stats, pred_values):
    """
    Обратная трансформация предсказаний в реальные калории.
    Работает и с numpy массивами, и с torch тензорами.

    Args:
        cal_stats: параметры нормализации калорий
        pred_values: предсказания модели (нормализованные)
    Returns:
        calories: значения в ккал
    """
    # 1. Обратная нормализация: (x * std) + mean
    calories = pred_values * cal_stats['std'] + cal_stats['mean']

    # 2. Обратный логарифм (если применялся log1p)
    if cal_stats['use_log']:
        calories = torch.expm1(calories)  # exp(x) - 1

    return calories


class CaloriesDataset(Dataset):
    def __init__(self, dish_df, config, image_transform=None, text_transform=None):
        super().__init__()
        self.mass_stats = {"mean": config.MASS_MEAN, "std": config.MASS_STD, "use_log": config.MASS_USE_LOG}
        self.cal_stats = {"mean": config.CAL_MEAN, "std": config.CAL_STD, "use_log": config.CAL_USE_LOG}

        # Загрузка справочника ингредиентов
        ingredients_df = pd.read_csv(config.INGREDIENTS_CSV_PATH, encoding='utf-8')
        self.ingr_map = dict(zip(ingredients_df['id'], ingredients_df['ingr']))

        # Копия dataframe для безопасной модификации
        self.dish_df = dish_df.copy()

        # Нормализации
        self.dish_df["total_mass"] = self._normalize_mass(self.dish_df['total_mass'])
        self.dish_df["total_calories"] = self._normalize_calories(self.dish_df['total_calories'])

        # Преобразует строку ингредиентов в список названий
        def parse_ingredients_list(ingr_string):
            # Разбиваем строку по ';', извлекаем ID и находим название в словаре
            return [
                self.ingr_map[int(ingr.strip().split('_')[1])]
                for ingr in ingr_string.split(';')
            ]

        # Предварительная обработка: названия ингредиентов и пути к изображениям
        self.dish_df['ingr_names_raw'] = self.dish_df['ingredients'].apply(parse_ingredients_list)
        self.dish_df['image_path'] = self.dish_df['dish_id'].apply(
            lambda x: config.IMAGES_DIR / x / "rgb.png"
        )

        # Сохранение трансформов
        self.image_transform = image_transform
        self.text_transform = text_transform

    def _normalize_mass(self, mass: pd.Series) -> pd.Series:
        if self.mass_stats["use_log"]:
            mass = np.log1p(mass)
        return (mass - self.mass_stats['mean']) / (self.mass_stats['std'] + 1e-8)

    def _normalize_calories(self, calories: pd.Series) -> pd.Series:
        if self.cal_stats["use_log"]:
            calories = np.log1p(calories)
        return (calories - self.cal_stats['mean']) / (self.cal_stats['std'] + 1e-8)

    def __len__(self):
        return len(self.dish_df)

    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]

        # Загрузка и трансформация изображения
        image = Image.open(row['image_path']).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image=np.array(image))["image"]

        # Применение текстовых трансформаций к ингредиентам
        ingr_raw = row['ingr_names_raw']
        if self.text_transform:
            ingr_list = [self.text_transform(ingr) for ingr in ingr_raw]
        else:
            ingr_list = ingr_raw

        return {
            "mass": row['total_mass'],
            "image": image,
            "ingr_list": ingr_list,
            "target": row['total_calories']
        }


def get_image_transforms(config, ds_type="train"):
    """
    Возвращает аугментации для изображений.

    Args:
        config: Конфигурация с параметрами модели
        ds_type: Тип датасета ("train" или "val"/"test")

    Returns:
        Compose объект с трансформациями
    """
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    input_height = cfg.input_size[1]
    input_width = cfg.input_size[2]

    if ds_type == "train":
        transforms = A.Compose(
            [
                # Подгонка размера
                A.SmallestMaxSize(
                    max_size=max(input_height, input_width),
                    p=1.0
                ),
                A.RandomCrop(
                    height=input_height,
                    width=input_width,
                    p=1.0
                ),

                # Геометрические аугментации
                A.Affine(
                    scale=(0.8, 1.2),                    # Scaling/Zoom
                    rotate=(-15, 15),                    # Rotation
                    translate_percent=(-0.1, 0.1),       # Translation (сдвиг)
                    shear=(-10, 10),                     # Perspective-like effect
                    fill=0,
                    p=0.8
                ),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Perspective(scale=(0.05, 0.1), p=0.3), # Perspective transform

                # Цветовые аугментации
                A.ColorJitter(
                    brightness=0.2,    # Brightness adjustment
                    contrast=0.2,      # Contrast adjustment
                    saturation=0.2,    # Saturation
                    hue=0.1,           # Hue shift
                    p=0.7
                ),

                # Качество изображения
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),   # Blur
                    A.Sharpen(alpha=(0.2, 0.5), p=0.5),         # Sharpening
                ], p=0.3),

                # Регуляризация
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(int(0.07 * input_height),
                                      int(0.15 * input_height)),
                    hole_width_range=(int(0.1 * input_width),
                                     int(0.15 * input_width)),
                    fill=0,
                    p=0.5
                ),

                # Нормализация и тензор
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )
    else:
        # Валидация/тест — только базовые трансформации
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(input_height, input_width),
                    p=1.0
                ),
                A.CenterCrop(
                    height=input_height,
                    width=input_width,
                    p=1.0
                ),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0)
            ]
        )

    return transforms

def get_text_transforms(ds_type="train"):
    """
    Возвращает аугментации для текста (ингредиенты).

    Args:
        config: Конфигурация с параметрами
        ds_type: Тип датасета ("train" или "val"/"test")

    Returns:
        Функция для применения текстовых аугментаций
    """
    if ds_type == "train":
        def text_augment(text):
            """
            Применяет аугментации к тексту.

            Args:
                text: Исходная строка

            Returns:
                Аугментированная строка
            """
            if not text or random.random() > 0.2:  # 20% шанс применения
                return text

            augmented_text = text
            operations = []

            # Keyboard Noise — замена на соседние клавиши
            if random.random() < 0.5:
                operations.append(_keyboard_noise)

            # Character Swap — перестановка символов
            if random.random() < 0.5:
                operations.append(_character_swap)

            if operations:
                # Применяем случайное количество операций (1 или 2)
                random.shuffle(operations)
                for op in operations[:random.randint(1, len(operations))]:
                    augmented_text = op(augmented_text)

            return augmented_text

        return text_augment
    else:
        # Валидация/тест — без аугментаций
        return lambda x: x

def _keyboard_noise(text):
    """
    Keyboard Noise: замена символов на соседние на клавиатуре.
    """
    if len(text) < 3:
        return text

    keyboard_layout = {
        'q': ['1', 'w', 'a', '2'], 'w': ['q', 'e', 'a', 's', '3'],
        'e': ['w', 'r', 's', 'd', '4'], 'r': ['e', 't', 'd', 'f', '5'],
        't': ['r', 'y', 'f', 'g', '6'], 'y': ['t', 'u', 'g', 'h', '7'],
        'u': ['y', 'i', 'h', 'j', '8'], 'i': ['u', 'o', 'j', 'k', '9'],
        'o': ['i', 'p', 'k', 'l', '0'], 'p': ['o', 'l', '-', '='],
        'a': ['q', 'w', 's', 'z'], 's': ['a', 'd', 'z', 'x', 'w'],
        'd': ['s', 'f', 'x', 'c', 'e'], 'f': ['d', 'g', 'c', 'v', 'r'],
        'g': ['f', 'h', 'v', 'b', 't'], 'h': ['g', 'j', 'b', 'n', 'y'],
        'j': ['h', 'k', 'n', 'm', 'u'], 'k': ['j', 'l', 'm', ',', 'i'],
        'l': ['k', 'p', ',', '.', 'o'], 'z': ['a', 's', 'x'],
        'x': ['z', 'c', 's', 'd'], 'c': ['x', 'v', 'd', 'f'],
        'v': ['c', 'b', 'f', 'g'], 'b': ['v', 'n', 'g', 'h'],
        'n': ['b', 'm', 'h', 'j'], 'm': ['n', ',', 'j', 'k'],
        ',': ['m', '.', 'k', 'l'], '.': [',', '/', 'l', ';'],
        '/': ['.', ';', 'l'], '-': ['p', '=', '0'], '=': ['-', 'p'],
        ';': ['l', '\'', 'p', '.'], '\'': [';', 'l', '/'],
    }

    result = list(text)
    num_changes = random.randint(1, max(1, len(text) // 5))

    for _ in range(num_changes):
        idx = random.randint(0, len(result) - 1)
        char = result[idx].lower()

        if char in keyboard_layout and random.random() < 0.3:
            result[idx] = random.choice(keyboard_layout[char])

    return ''.join(result)

def _character_swap(text):
    """
    Character Swap: перестановка соседних символов (опечатки).
    """
    if len(text) < 2:
        return text

    result = list(text)
    num_swaps = random.randint(1, max(1, len(text) // 7))

    for _ in range(num_swaps):
        idx = random.randint(0, len(result) - 2)
        # Меняем местами соседние символы
        result[idx], result[idx + 1] = result[idx + 1], result[idx]

    return ''.join(result)

def prepare_dataloaders(config):
    dish_df = pd.read_csv(config.DISH_CSV_PATH, encoding='utf-8')

    # val - 15% от общих данных, также как тест
    val_count = len(dish_df[dish_df["split"] == "test"])
    train_count = len(dish_df[dish_df["split"] == "train"]) - val_count

    dish_train_val_df = dish_df[dish_df["split"] == "train"]
    dish_df_train, dish_df_val = train_test_split(
        dish_train_val_df,
        train_size=train_count,
        test_size=val_count,
        random_state=42,
        shuffle=True
    )

    # Сброс индекса для быстрого доступа по iloc
    dish_train_df = dish_df_train.reset_index(drop=True)
    dish_val_df = dish_df_val.reset_index(drop=True)
    dish_test_df = dish_df[dish_df["split"] == "test"].reset_index(drop=True)

    train_image_transform = get_image_transforms(config, "train")
    val_image_transform = get_image_transforms(config, "val")
    test_image_transform = get_image_transforms(config, "test")

    train_text_transform = get_text_transforms("train")

    train_dataset = CaloriesDataset(dish_train_df,
                                    config,
                                    train_image_transform,
                                    train_text_transform)

    val_dataset = CaloriesDataset(dish_val_df,
                                  config,
                                  val_image_transform
                                  )

    test_dataset = CaloriesDataset(dish_test_df,
                                   config,
                                   test_image_transform
                                   )

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4
    )

    return train_dataloader, val_dataloader, test_dataloader

def collate_fn(batch, tokenizer, max_length=128):
    # 1. Собираем все ингредиенты и считаем их количество
    all_ingrs = []
    ingr_counts = []

    for item in batch:
        ingr_list = item["ingr_list"]
        ingr_counts.append(len(ingr_list))
        all_ingrs.extend(ingr_list)

    # 2. Токенизация с динамическим паддингом
    encoded = tokenizer(
        all_ingrs,
        truncation=True,
        max_length=max_length,
        padding='longest',
        return_tensors='pt'
    )

    # 3. Получаем фактическую длину после токенизации
    batch_size = len(batch)
    max_ingrs = max(ingr_counts)
    seq_len = encoded['input_ids'].shape[1]  # Динамическая длина (например, 12, а не 50)

    # 4. Создаём тензоры под фактический размер
    ingrs = torch.zeros((batch_size, max_ingrs, seq_len), dtype=torch.long)
    ingr_masks = torch.zeros((batch_size, max_ingrs, seq_len), dtype=torch.long)

    # 5. Заполняем данными
    idx = 0
    for i, count in enumerate(ingr_counts):
        ingrs[i, :count, :] = encoded['input_ids'][idx:idx + count]
        ingr_masks[i, :count, :] = encoded['attention_mask'][idx:idx + count]
        idx += count

    # 6. Остальные данные
    images = torch.stack([item["image"] for item in batch])
    targets = torch.FloatTensor([item["target"] for item in batch])
    masses = torch.FloatTensor([item["mass"] for item in batch])

    return {
        "ingrs": ingrs,
        "attention_mask": ingr_masks,
        "image": images,
        "mass": masses,
        "target": targets,
    }
