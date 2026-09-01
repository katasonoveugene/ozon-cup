# Решение задачи «Контроль качества»

В репозитории находятся код для запуска и обучения, готовые модели и артефакты
решения, набравшего `0.7547852474323062` на public leaderboard.

## Описание метода

Решение объединяет текстовую модель и классификатор на мультимодальных
эмбеддингах. Модели и пороги обучаются отдельно для категорий «БАД» и
«Легковоспламеняющиеся».

Текстовая часть строит два TF-IDF-представления карточки:

- словные n-граммы длины 1–2;
- символьные n-граммы длины 3–6.

Название товара в исходном тексте повторяется дважды, после чего словные и
символьные признаки объединяются и передаются в `LinearSVC`. Для каждого из
семи начальных состояний строятся модели на пяти фолдах. Их оценки усредняются
по фолдам внутри одного состояния, а затем для каждой строки берётся медиана
по семи состояниям.

Мультимодальное представление строится моделью
`Qwen/Qwen3-VL-Embedding-2B`. На вход передаются название, описание, категория
и изображения товара. Для категории «БАД» поверх нормализованных эмбеддингов
обучается SVM с RBF-ядром на пяти фолдах.

Перед объединением оценки компонентов стандартизуются отдельно внутри каждой
категории. Для «БАД» итоговая оценка состоит из равных долей текстового и
мультимодального компонентов. Для «Легковоспламеняющихся» используется только
текстовый компонент. Порог классификации также подбирается отдельно для каждой
категории.

После классификации применяются два детерминированных уточнения: точное
совпадение нормализованного описания с однозначно размеченными примерами
обучающей выборки и небольшой набор консервативных правил по тексту карточки.

По условию задачи каждая строка результата содержит вердикт и пояснение длиной
от 50 до 300 символов. Сначала ансамбль фиксирует класс товара, после чего
`Qwen/Qwen3.5-4B` формирует пояснение по названию, описанию и категории.
Генерация текста не участвует в классификации и не может изменить вердикт.

## Основные файлы

- `run.py` — точка входа для получения предсказаний;
- `train_sparse_bag.py` — обучение текстового ансамбля;
- `train_dense_bag.py` — обучение классификаторов на мультимодальных
  эмбеддингах;
- `artifacts/sparse_bag.joblib` и `artifacts/dense_bag.joblib` — готовые
  классификаторы;
- `artifacts/fusion.json` — веса, параметры стандартизации и пороги итоговой
  модели;
- `outputs/mixed.npy` — мультимодальные эмбеддинги обучающей выборки;
- `outputs/sparse_bag_oof.npz` и `outputs/dense_bag_oof.npz` — вневыборочные
  оценки компонентов;
- `outputs/duplicate_bag_features.npz` — признаки точных совпадений,
  рассчитанные на обучающих фолдах;
- `experiments/fusion_search.py` — подбор весов и порогов итоговой модели;
- `quality_core/` — реализация моделей, обработки данных и формирования
  результата;
- `third_party/` — локальные копии вспомогательного кода Qwen с лицензиями и
  метаданными исходных пакетов.

## Подготовка после клонирования

Готовые модели и матрица эмбеддингов хранятся через Git LFS:

```bash
git lfs install
git lfs pull
```

## Получение предсказаний

Среда выполнения указана в `metadata.json`:

```text
odsai/ecup26-quality-baseline:1.0
```

В каталоге `SHARED_MODELS_PATH` должны находиться модели:

```text
Qwen/Qwen3-VL-Embedding-2B
Qwen/Qwen3.5-4B
```

Если переменная `SHARED_MODELS_PATH` не задана, используется каталог
`/shared_models`. В тестовом CSV обязательны столбцы `id`, `name`,
`description` и `category`. Рядом с CSV должен находиться каталог `images`:

```text
input/
├── test.csv
└── images/
```

Команда запуска:

```bash
python -u run.py \
  --test-data-path input/test.csv \
  --output-path output/submission.csv
```

Выходной CSV содержит столбцы `id,result`. Значение `result` имеет один из
форматов:

```text
<комментарий>...<вердикт>бан
<комментарий>...<вердикт>не бан
```

## Воспроизведение обучения

Для повторного обучения нужен официальный файл `data/data.csv`. Точная матрица
мультимодальных эмбеддингов, выровненная по строкам этого файла, уже находится
в `outputs/mixed.npy`.

Текстовый ансамбль:

```bash
python -u train_sparse_bag.py \
  --data data/data.csv \
  --output outputs/rebuilt/sparse_bag.joblib \
  --oof-output outputs/rebuilt/sparse_bag_oof.npz \
  --report outputs/rebuilt/sparse_bag_report.json \
  --seeds 42,3407,1337,2025,20260829,20260701,777 \
  --jobs 5
```

Модель на эмбеддингах:

```bash
python -u train_dense_bag.py \
  --data data/data.csv \
  --embeddings outputs/mixed.npy \
  --selection artifacts/kernel_selection.json \
  --output outputs/rebuilt/dense_bag.joblib \
  --oof-output outputs/rebuilt/dense_bag_oof.npz \
  --report outputs/rebuilt/dense_bag_report.json \
  --seeds 20260829 \
  --jobs 5
```

Признаки точных совпадений можно заново получить из официальных данных и
зафиксированных оценок семи текстовых моделей:

```bash
PYTHONPATH=. python -u experiments/duplicate_bag_cv.py \
  --data data/data.csv \
  --images-zip data/images.zip \
  --base-oof outputs/raw_sparse_hybrid_7seed_oof.npz \
  --output outputs/rebuilt/duplicate_bag_features.npz
```

Повторный подбор весов и порогов:

```bash
PYTHONPATH=. python -u experiments/fusion_search.py \
  --data data/data.csv \
  --sparse-oof outputs/sparse_bag_oof.npz \
  --dense-oof outputs/dense_bag_oof.npz \
  --duplicate-features outputs/duplicate_bag_features.npz \
  --output outputs/rebuilt/fusion_search.json \
  --config-output outputs/rebuilt/fusion.json
```

Значения `embedding_batch_size` и `comment_batch_size` в готовом
`artifacts/fusion.json` подобраны под среду проверки и влияют только на размер
пакета при инференсе. Веса компонентов, параметры стандартизации и пороги
совпадают с результатом `fusion_search.py`.

## Проверка артефактов

| Файл | Размерность или размер | SHA256 |
| --- | ---: | --- |
| `outputs/mixed.npy` | `12971 x 2048`, float16 | `cea52432aa2d61888a81b5ad7548322beac86c50eb1211300d70ac2fe773a06f` |
| `outputs/raw_sparse_hybrid_7seed_oof.npz` | 12 971 строк | `bfecca3de38aec8885294d601faaafed39d31e49b45bca8199f71ea3c773aaa2` |
| `outputs/duplicate_bag_features.npz` | 12 971 строк, 7 наборов OOF, 44 признака | `5589b2af0d396fac855c2a5f9081900423d8bf626d6df02d610bd518f8d43354` |
| `artifacts/sparse_bag.joblib` | 166 815 799 байт | `f682eec152acff3058716c69ec0f35fb91873fce26c255382abfb403b0e53410` |
| `artifacts/dense_bag.joblib` | 59 260 763 байта | `b0d5280cf1f44a415b3acac4b5c03cfd78c4b341ec11fdcdbc78f8a67a81f160` |
| `artifacts/fusion.json` | параметры объединения | `3ebf9b3b93f15a02864d9eda8e2139ff150f117edd0874e73f39c467ef7030ec` |
| `artifacts/kernel_selection.json` | параметры классификаторов | `fcbda8b2c5a53377ec6e15c3ad0f5ce6d7183f2a861e79f43ac0b46bb279fe51` |

## Модели и данные

Для мультимодальных эмбеддингов используется `Qwen/Qwen3-VL-Embedding-2B`, а
для пояснений при получении предсказаний — `Qwen/Qwen3.5-4B`. Обе модели
распространяются по лицензии Apache-2.0. Их размеры — 2B и 4B соответственно,
что соответствует ограничению задачи.

Для обучения не использовались сгенерированные метки, псевдоразметка,
синтетические примеры, проприетарные API или внешние размеченные данные.
