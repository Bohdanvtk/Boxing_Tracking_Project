# Results API — посібник з використання

`boxing_project.results` — це легка read-only обгортка над `observations.parquet`
(тільки numpy + pandas). Вона перетворює таблицю спостережень у впорядковані
часові послідовності bbox / BODY_25 keypoints / metadata для конкретного боксера.

> Усі приклади передбачають, що inference вже відпрацював і створив
> `…/dataset/observations.parquet`. Сам модуль нічого не пише на диск.

---

## 0. Встановлення та імпорт

```bash
pip install -r requirements/results.txt   # numpy, pandas, pyarrow
```

```python
from boxing_project.results import BoxingResults
```

Якщо потрібні класи для type hints / isinstance:

```python
from boxing_project.results import (
    BoxingResults,
    TrackSelection,
    BoxerSegment,
    SegmentCollection,
    SegmentData,
    FrameObservation,
    BBox,
    AmbiguousObservationError,
)
```

---

## 1. Завантаження даних

```python
# Варіант А: шлях до output-папки (шукає dataset/observations.parquet)
results = BoxingResults("data/output/test")

# Варіант Б: прямий шлях до parquet
results = BoxingResults("data/output/test/dataset/observations.parquet")

# Варіант В: parquet просто в корені папки (fallback)
results = BoxingResults("some/dir")  # шукає some/dir/observations.parquet
```

Parquet читається **один раз** і кешується в пам'яті.

---

## 2. Огляд того, що є в датасеті

```python
results.available_global_ids   # напр. [1, 2]
results.available_epochs       # напр. [2, 6, 8]
results.path                   # Path до фактичного parquet
len(results)                   # кількість рядків
```

---

## 3. Один боксер на одному кадрі (`FrameObservation`)

```python
obs = (
    results
    .global_id(1)
    .epoch(6)
    .frame(449)
)

obs.frame_idx        # 449
obs.bbox             # BBox(x1, y1, x2, y2)
obs.kps              # np.ndarray (25, 3) -> [x, y, confidence]
obs.meta             # dict з tracking-метаданими + n_visible_keypoints

obs.is_observed      # чи існує рядок у parquet
obs.has_keypoints    # чи є реальні keypoints
obs.has_detection    # matched-детекція + keypoints
```

Робота з bbox:

```python
bb = obs.bbox
bb.x1, bb.y1, bb.x2, bb.y2
bb.width, bb.height
import numpy as np
np.asarray(bb)       # (4,) -> [x1, y1, x2, y2]
bb.to_numpy()        # те саме
```

---

## 4. Сегмент: послідовність кадрів одного боксера (`BoxerSegment`)

Є два способи вибрати діапазон — і вони мають різну семантику.

### `frames(start, end)` — тільки наявні кадри

Повертає рядки, які **реально існують** у `[start, end]` (включно). Довжина
може бути меншою за `end - start + 1`.

```python
segment = (
    results
    .global_id(1)
    .epoch(6)
    .frames(444, 463)
)
len(segment)              # напр. 17 (а не 20), якщо 3 кадри відсутні
segment.observation_mask  # усі True — padding немає
```

### `window(start_frame, length)` — фіксована довжина з padding

Повертає **рівно `length`** часових позицій. Відсутні кадри доповнюються
порожніми позиціями (bbox = NaN, keypoints = NaN/0). Саме це потрібно для входу
нейромережі.

```python
segment = (
    results
    .global_id(1)
    .epoch(6)
    .window(start_frame=444, length=20)
)

segment.frames.shape            # (20,)
segment.bbox.shape              # (20, 4)
segment.kps.shape               # (20, 25, 3)
segment.observation_mask.shape  # (20,)
segment.detection_mask.shape    # (20,)
```

---

## 5. Дані сегмента та маски

```python
segment.frames            # (T,)      — індекси кадрів
segment.bbox              # (T, 4)     — [x1, y1, x2, y2]
segment.kps               # (T, 25, 3) — [x, y, confidence]
segment.meta              # DataFrame  — по рядку metadata на кадр

segment.observation_mask  # (T,) True там, де є рядок у parquet
segment.detection_mask    # (T,) True там, де matched-детекція + keypoints
segment.keypoints_mask    # (T,) True там, де є хоч одна keypoint з conf > 0
```

Зручний контейнер усього одразу:

```python
data = segment.data
data.frames, data.bbox, data.kps, data.meta
data.observation_mask, data.detection_mask
```

Маска по окремих суглобах через confidence:

```python
joint_mask = segment.kps[..., 2] > 0.3   # (T, 25) bool
```

---

## 6. Ітерація по сегменту

```python
for obs in segment:                 # кожен елемент — FrameObservation
    print(obs.frame_idx, obs.is_observed, obs.has_detection)
    print(obs.bbox, obs.kps.shape)
```

---

## 7. Кілька боксерів одночасно

Вибірка з кількома global ID — це `TrackSelection`, а не один сегмент.

```python
selection = (
    results
    .global_ids([1, 2])
    .epoch(6)
    .frames(444, 560)
)
type(selection)          # TrackSelection
```

Розбити на сегменти по боксерах:

```python
segments = selection.segments()      # SegmentCollection
boxer_1 = segments[1]
boxer_2 = segments[2]
print(boxer_1.kps.shape, boxer_2.kps.shape)

segments.global_ids                  # [1, 2]
1 in segments                        # True
for seg in segments:
    print(seg.global_id, seg.frames.shape, seg.kps.shape)
```

`window()` для кількох боксерів повертає `SegmentCollection` (по одному
padded-сегменту на боксера, спільна вісь):

```python
coll = results.global_ids([1, 2]).epoch(6).window(start_frame=444, length=20)
coll[1].kps.shape    # (20, 25, 3)
coll[2].kps.shape    # (20, 25, 3)
```

---

## 8. Фрагменти local track одного global ID

Один боксер може складатися з кількох `(epoch_id, local_track_id)` фрагментів.

```python
table = results.global_id(1).local_tracks()
# columns: global_track_id, epoch_id, local_track_id,
#          start_frame, end_frame, observations, matched_frames
print(table)
```

Вибрати конкретний local track (id унікальний лише в межах epoch):

```python
sel = results.local_track(local_track_id=2, epoch_id=6)
```

---

## 9. Базові фільтри

```python
results.global_id(1)                 # один global ID
results.global_ids([1, 2])           # кілька
results.epoch(6)                     # одна epoch
results.epochs([5, 6])               # кілька

results.at_frame(500)                # усі боксери на кадрі 500 (TrackSelection)
results.frame(500)                   # один FrameObservation (вимагає 1 рядок)

results.matched_only()               # is_matched == True
results.confirmed_only()             # confirmed == True
results.with_keypoints()             # тільки рядки з keypoints
results.unassigned()                 # global_track_id is null
```

Фільтри ланцюгуються (кожен повертає новий view, джерело не змінюється):

```python
view = results.global_id(1).epoch(6).matched_only().confirmed_only()
```

---

## 10. Комбінований фільтр `select()`

Скаляри і списки приймаються однаково:

```python
selection = results.select(
    global_ids=[1, 2],
    epoch_ids=6,
    local_track_ids=[1, 2],
    frame_range=(444, 560),
    matched=True,
    confirmed=True,
)
```

---

## 11. Випадкові спостереження для кожного боксера

```python
samples = results.sample_per_global(
    n=5,
    random_state=42,
    matched_only=True,
)
# до 5 рядків на кожен ненульовий global_track_id; null пропускається
samples.df.groupby("global_track_id").size()
```

Корисно для побудови crop-датасетів, перевірки якості, візуального аудиту.

---

## 12. Вихід у «сирий» pandas

Коли API чогось не покриває:

```python
results.df                 # внутрішній DataFrame (raw, можна зіпсувати state)
selection.df
segment.df

results.to_pandas()        # незалежна копія (за замовчуванням copy=True)
segment.to_pandas(copy=False)   # без копії, якщо точно read-only
```

Далі — звичайний pandas:

```python
df = results.to_pandas()
df[df.e_app_coverage > 0.8].groupby("global_track_id").size()
```

---

## 13. Підготовка батча для моделі

Один боксер, одне фіксоване вікно:

```python
segment = results.global_id(1).epoch(6).window(start_frame=444, length=32)
x = segment.kps              # (32, 25, 3)
mask = segment.detection_mask
prediction = model.predict(x)
```

Батч із кількох вікон (sliding window):

```python
import numpy as np

starts = range(444, 544, 16)        # крок 16
clips, masks = [], []
for s in starts:
    seg = results.global_id(1).epoch(6).window(start_frame=s, length=32)
    clips.append(seg.kps)
    masks.append(seg.detection_mask)

batch = np.stack(clips)              # (N, 32, 25, 3)
batch_mask = np.stack(masks)         # (N, 32)
```

Батч по кількох боксерах за один кадровий діапазон:

```python
segs = results.global_ids([1, 2]).epoch(6).window(444, 32)
batch = np.stack([segs[g].kps for g in segs.global_ids])   # (2, 32, 25, 3)
```

---

## 14. Як модель має враховувати padding і пропуски

Дві різні ситуації кодуються масками:

| Ситуація | observation_mask | detection_mask | bbox | keypoints |
|---|---|---|---|---|
| Реальна детекція | True | True | реальний | реальні |
| Трек живий, детекції немає (Kalman) | True | False | прогноз | NaN / conf 0 |
| Кадру взагалі немає (padding у window) | False | False | NaN | NaN / conf 0 |

Рекомендований патерн — не «вигадувати» координати, а передавати маску в модель:

```python
seg = results.global_id(1).epoch(6).window(444, 32)
x = np.nan_to_num(seg.kps, nan=0.0)     # NaN -> 0 для тензора
mask = seg.detection_mask               # де keypoints справжні
# подавайте mask у модель (напр. attention mask / loss weighting)
```

Bbox для кадрів без детекції — це валідний прогноз треку, його можна
використовувати (напр. для crop), орієнтуючись на `observation_mask`.

---

## 15. Обробка помилок

### Неоднозначність

Якщо одному `(global_track_id, frame_idx)` відповідає кілька рядків (кілька
local tracks зіставлено з одним боксером на кадрі), API не вибирає мовчки
перший:

```python
try:
    obs = results.global_id(1).epoch(6).frame(449)
except AmbiguousObservationError as e:
    print(e.global_track_id, e.frame_idx)
    print(e.rows[["epoch_id", "local_track_id"]])
```

Помилка виникає і ліниво — при доступі до масивів сегмента:

```python
results.global_id(1).window(444, 20).kps   # може кинути AmbiguousObservationError
```

Кілька **різних** global ID на одному кадрі — це нормально для загальної
вибірки, але `frame()` тоді просить звузити:

```python
results.epoch(6).frame(449)   # ValueError: Multiple global ids ... use global_id(...)
```

### Валідація аргументів

```python
results.global_id(1).window(444, 0)     # ValueError: length must be greater than zero
results.global_id(1).frames(500, 400)   # ValueError: end must be >= start
results.sample_per_global(n=0)          # ValueError: n must be greater than zero
```

### Валідація даних

```python
BoxingResults("dir/without_columns")    # ValueError: Missing required columns: [...]
# неправильна кількість keypoints -> ValueError: Expected 25 BODY_25 keypoints, received N
```

---

## 16. Типовий повний сценарій

```python
from boxing_project.results import BoxingResults
import numpy as np

results = BoxingResults("data/output/test")

# 1) Хто є в датасеті
for gid in results.available_global_ids:
    print(gid, results.global_id(gid).local_tracks())

# 2) Беремо одного боксера, фіксоване вікно під модель
segment = (
    results
    .global_id(1)
    .epoch(6)
    .window(start_frame=444, length=32)
)

x = np.nan_to_num(segment.kps, nan=0.0)   # (32, 25, 3)
mask = segment.detection_mask             # (32,)

# 3) Прогноз
prediction = model.predict(x)             # ваша модель

# 4) Аналіз пропусков
print("кадрів усього:", len(segment))
print("з детекцією:", int(segment.detection_mask.sum()))
print("padding:", int((~segment.observation_mask).sum()))
```

---

## Шпаргалка повернень

| Виклик | Що повертає |
|---|---|
| `global_id`, `epoch`, `local_track`, `at_frame`, `matched_only`, `select`, … | `TrackSelection` |
| `frame(n)` | `FrameObservation` (рівно 1 рядок) |
| `frames(a, b)` — 1 global id | `BoxerSegment` |
| `frames(a, b)` — 0 / кілька global id | `TrackSelection` |
| `window(s, L)` — 1 global id | `BoxerSegment` (рівно L, з padding) |
| `window(s, L)` — кілька global id | `SegmentCollection` |
| `segments()` | `SegmentCollection` |
| `local_tracks()` | `pandas.DataFrame` |
| `sample_per_global()` | `TrackSelection` |
