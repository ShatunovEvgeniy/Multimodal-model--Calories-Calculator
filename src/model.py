import timm
from transformers import AutoModel
import torch
import torch.nn as nn

from src.dataset import denormalize_calories

class CaloriesModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.cal_stats = {"mean": config.CAL_MEAN, "std": config.CAL_STD, "use_log": config.CAL_USE_LOG}

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        self.ingr_attention = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 4),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_DIM // 4, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, ingrs, attention_mask, image, mass):
        """
        ingrs: [B, N, L] - batch, num_ingredients, tokens_per_ingredient
        attention_mask: [B, N, L] - маска токенов
        image: [B, C, H, W]
        mass: [B]
        """
        batch_size, num_ingredients, seq_len = ingrs.shape

        # === 1. Обработка каждого ингредиента ===
        ingrs_flat = ingrs.view(-1, seq_len)  # [B*N, L]
        mask_flat = attention_mask.view(-1, seq_len)  # [B*N, L]

        # [B*N, L] -> [B*N, hidden_size]
        text_outputs = self.text_model(ingrs_flat, attention_mask=mask_flat)
        ingredient_embs = text_outputs.last_hidden_state[:, 0, :]  # [CLS]

        # Восстанавливаем: [B*N, H] -> [B, N, H]
        ingredient_embs = ingredient_embs.view(batch_size, num_ingredients, -1)

        # === 2. Projection в общее пространство до attention ===
        ingredient_embs_proj = self.text_proj(ingredient_embs)  # [B, N, HIDDEN_DIM]

        # === 3. Attention-weighted pooling ===
        # Считаем "важность" каждого ингредиента
        attn_scores = self.ingr_attention(ingredient_embs_proj).squeeze(-1)  # [B, N]

        # Маскируем паддинг-ингредиенты: если все токены ингредиента — паддинг, вес = -inf
        ingr_valid_mask = attention_mask.any(dim=-1)  # [B, N], bool
        attn_scores = attn_scores.masked_fill(~ingr_valid_mask, -1e9)

        # Softmax по ингредиентам -> веса в [0, 1], сумма = 1
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N]

        # Взвешенная сумма эмбеддингов
        text_features = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, N]
            ingredient_embs_proj  # [B, N, HIDDEN_DIM]
        ).squeeze(1)  # [B, HIDDEN_DIM]

        # === 4. Обработка изображения и массы ===
        image_features = self.image_model(image)  # [B, image_hidden_size]

        image_emb = self.image_proj(image_features)  # [B, HIDDEN_DIM]
        numeric_emb = self.mass_proj(mass.unsqueeze(1))  # [B, 1] -> [B, HIDDEN_DIM]

        # === 5. Fusion модальностей ===
        fused_emb = text_features + image_emb + numeric_emb  # [B, HIDDEN_DIM]

        # === 6. Предсказание ===
        calories = self.head(fused_emb).squeeze(-1)  # [B]
        return calories

    @torch.no_grad()
    def infer(
            self,
            ingrs: torch.Tensor,
            attention_mask: torch.Tensor,
            image: torch.Tensor,
            mass: torch.Tensor | float | int,
            device: torch.device | str | None = None,
            return_dict: bool = False,
            clip_negative: bool = True,
    ) -> torch.Tensor | dict:
        """
        Инференс модели для предсказания калорийности.

        Args:
            ingrs: Tensor [B, N, L] - токенизированные ингредиенты
            attention_mask: Tensor [B, N, L] - маска токенов
            image: Tensor [B, C, H, W] - предобработанное изображение
            mass: Tensor [B], float или int - масса блюда в граммах
            device: Устройство для инференса (None = auto)
            return_dict: Если True, возвращает dict с метаданными
            clip_negative: Если True, обрезает отрицательные предсказания до 0

        Returns:
            Tensor [B] с предсказанными калориями ИЛИ dict с результатами
        """
        # Сохраняем и переключаем режим
        training = self.training
        self.eval()

        try:
            # Авто-определение устройства
            if device is None:
                device = next(self.parameters()).device

            # Нормализация mass к тензору [B]
            if isinstance(mass, (int, float)):
                mass = torch.tensor([mass], device=device)
            elif isinstance(mass, torch.Tensor) and mass.dim() == 0:
                mass = mass.unsqueeze(0)

            # Перенос на device
            ingrs = ingrs.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            mass = mass.to(device, non_blocking=True).float()

            # Forward pass
            calories_pred = self.forward(
                ingrs=ingrs,
                attention_mask=attention_mask,
                image=image,
                mass=mass
            )
            calories_pred = denormalize_calories(self.cal_stats, calories_pred)

            if return_dict:
                result = {
                    'calories': calories_pred,
                    'input_mass': mass,
                }
                if clip_negative:
                    result['calories_clipped'] = torch.clamp(calories_pred, min=0)
                return result

            return torch.clamp(calories_pred, min=0) if clip_negative else calories_pred

        finally:
            # Восстанавливаем режим обучения
            if training:
                self.train()
