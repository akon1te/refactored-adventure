# Arcane Adventure

Проект для дообучения и генерации изображений в стиле "arcane" на основе пайплайна Stable Diffusion Img2Img.

## Использованная модель
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

## Структура проекта

- **.gitignore** – файлы и папки, исключаемые из репозитория.
- **app/** – веб-приложение на [Gradio](https://gradio.app/) для генерации изображений.  
  Основной файл: [`app/app.py`](app/app.py)
- **data/** – данные для обучения, включая изображения и инструкцию по скачиванию датасета.  
  Файл с инструкцией: [`data/README.md`](data/README.md)
- **logs/** – логи обучения и контрольные точки модели.
- **scripts/** – скрипты для запуска обучения и инференса.  
  - Скрипт обучения: [`scripts/run_train.sh`](scripts/run_train.sh)  
  - Скрипт инференса: [`scripts/infer.sh`](scripts/infer.sh)
- **src/** – исходный код для обучения и инференса.
  - Скрипт обучения: [`src/train.py`](src/train.py)
  - Скрипт инференса: [`src/inference.py`](src/inference.py)

## Требования

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)
- torchvision
- [Pillow](https://python-pillow.org/)
- Gradio
- tqdm
- peft
- datasets

Установить зависимости можно командой:

```sh
pip install torch torchvision diffusers pillow gradio tqdm peft datasets
```

## Тренировка модели

Запустите тренировку модели, выполнив скрипт:

```sh
bash scripts/run_train.sh
```

Параметры обучения задаются через аргументы командной строки в файле [`src/train.py`](src/train.py).

## Инференс

Для генерации изображений из дообученной модели запустите скрипт:

```sh
bash scripts/infer.sh
```

Параметры инференса задаются через аргументы командной строки в файле [`src/inference.py`](src/inference.py).

## Веб-приложение

Для запуска веб-интерфейса на базе Gradio выполните:

```sh
python app/app.py
```

Интерфейс позволяет вводить текстовый запрос, загружать изображение и настраивать параметры генерации.

## Датасет

Для обучения используется датасет, скачанный с [Kaggle](https://www.kaggle.com/datasets/artermiloff/arcanefaces). Подробная инструкция по скачиванию находится в файле [`data/README.md`](data/README.md).

## Логи

Логи обучения и контрольные точки модели сохраняются в папке `logs/arcane_finetune`. Здесь можно отслеживать процесс обучения и находить файлы с контрольными точками модели.
