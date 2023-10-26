# ASR project barebones

## Installation guide

```shell
pip install -r ./requirements.txt
```

запуск трейна

```
python hw_asr/train.py -c hw_asr/hw_asr/configs/final.json
```

После этого надо переложить из папки saved последний чекпоинт и его config в папку `default_test_model`, для запуска теста

запуск теста

```
python hw_asr/test.py
```
