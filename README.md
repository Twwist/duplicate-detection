# Duplicate detection
Создание сервиса по определению дубликатов видео, публикуемых пользователями.

## Команда to2to models
Состав:

- Муратшин Динияр [@iztwwist](https://t.me/iztwwist)
- Гарифуллин Тимур [@Murkat07](https://t.me/Murkat07)
- Мурзаев Михаил [@mishkashishka333](https://t.me/mishkashishka333)

## Описание решения
Дубликаты видео проверяются на соответствие аудио, тексту и кадрам с другими видео из векторной базы данных.  
Схожесть видео определяется с помощью косинусного расстояния между эмбеддингами видео, аудио и текста.

## Фичи решения
- Быстрая и масштабируемая векторная база данных Qdrant
- Сравнение нескольких кадров видео
- Модульный пайплайн
- Мультимодальность

## Пайплайны
Пайплайн - это комбинация экстрактора и энкодера, который преобразует входные данные в векторные данные.
### Экстракторы
Экстракторы достают контент из файлов, являются генераторами.
Реализованные экстракторы:
* Видео:
    - VideoEveryNFramesExtractor - получение кадров видео, позиция которых кратна n
    - VideoNEvenlySpacedExtractor - получение n кадров видео, распределенных равномерно
    - VideoKeyFrameFFmpegExtractor - получение ключевых кадров видео с использованием ffmpeg
* Аудио:
    - AudioFull - получение всего аудио
    - AudioNSecondsSplit - получение аудио с разделением на чанки по n секунд
* Текст:
    - TextExtractorWhisperx - транскибирование текста с помощью WhisperX

### Энкодеры
Энкодеры преобразуют объекты в векторный вид.
Реализованные энкодеры:
* Видео:
    - ColorHistogramEncoder - цветовая гистограмма (распределение цветов в изображении)
    - TimmEncoder - модели из репозитория [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models)
    - CLIPEncoder - модель архитектуры CLIP
* Аудио:
    - Wav2Vec2Encoder - модель архитектуры Wav2Vec2
    - EnCodecEncoder - модель архитектуры EnCodec
* Текст:
    - TextEncoderE5: модель архитектуры E5

Пайплайны можно создавать вручную:
```python
from extractors import *
from encoders import *
from general import Pipeline

pipeline = Pipeline(VideoEveryNFramesExtractor(n=5), TimmEncoder("vit_base_patch16_224"), pool=False)
video_embeddings = pipeline(video_path)
```
А можно считывать из yaml-файла:
```python
from general import Pipeline

pipeline = Pipeline.from_yaml("video_pipeline.yaml")
video_embeddings = pipeline(video_path)
```
Пример yaml-файла:
```yaml
extractor:
  name: VideoEveryNFramesExtractor
  args:
    n: 5

encoder:
  name: TimmEncoder
  args:
    name: vit_base_patch16_224
    device: cuda

pool: false
```
## Установка
Используя conda:
```bash
$ git clone https://github.com/Twwist/duplicate-detection.git
$ cd duplicate-detection
$ cp example.env .env
$ conda env create -n duplicate-detection -f environment.yml
```

## Использование
Запускаем сервис:
```bash
$ conda activate duplicate-detection
$ python api.py
```
