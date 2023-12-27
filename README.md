---
tags:
- audio-classification
- speechbrain
- Emotion
- Recognition
- wav2vec2
- pytorch
datasets:
- iemocap
metrics:
- Accuracy 75%
---
WORKING PROTOTYPE -> [https://immelstorun.github.io/speech_emo_recognition/](https://immelstorun.github.io/speech_emo_recognition/)
# MIPT prototype Emotion Recognition with wav2vec2 base on IEMOCAP

`MIPT students prototype for speech emotion recognition using Machine Learning pretrained wav2vec model on IEMOCAP dataset powered by speechbrain toolkit`

<p align="center">
  <img src="about.png" width="80%" height="80%" alt="О проекте">
</p>

Шаги реализации проекта

1.	Настройка и интеграция предобученной модели wav2vec2 из базы предобученных моделей от Speechbrain. [Модель](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
2.	Создание простого пользовательского интерфейса для загрузки и воспроизведения аудиозаписей на интерфейсе Gradio
3.	Реализация функции классификации эмоций на основе загруженных аудио, развернув экземпляр в HuggingFace.co [Контейнер с работающим приложением](https://huggingface.co/spaces/immelstorun/speech_emotion_detection)
4.	Организация возвращения результатов классификации пользователю.
5.	Подготовка демонстрационного стенда с возможностью тестирования системы через Github.io [Перейти на сайт прототипа](https://immelstorun.github.io/speech_emo_recognition/)

### Быстрый запуск в Google Colab

An external `py_module_file=custom.py` is used as an external Predictor class into this HF repos. We use `foreign_class` function from `speechbrain.pretrained.interfaces` that allow you to load you custom model. 

```python
from speechbrain.pretrained.interfaces import foreign_class
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
print(text_lab)
```
 The prediction tensor will contain a tuple of (embedding, id_class, label_name).

This repository provides all the necessary tools to perform emotion recognition with a fine-tuned wav2vec2 (base) model using SpeechBrain. 
It is trained on IEMOCAP training data.


For a better experience, we encourage you to learn more about
[SpeechBrain](https://speechbrain.github.io). The model performance on IEMOCAP test set is:

| Release | Accuracy(%) | 
|:-------------:|:--------------:|
| 19-10-21 | 78.7 (Avg: 75.3) | 


## Pipeline description

This system is composed of an wav2vec2 model. It is a combination of convolutional and residual blocks. The embeddings are extracted using attentive statistical pooling. The system is trained with Additive Margin Softmax Loss.  Speaker Verification is performed using cosine distance between speaker embeddings.

The system is trained with recordings sampled at 16kHz (single channel).
The code will automatically normalize your audio (i.e., resampling + mono channel selection) when calling *classify_file* if needed.


## Install SpeechBrain

First of all, please install the **development** version of SpeechBrain with the following command:

```
pip install speechbrain
```

Please notice that we encourage you to read our tutorials and learn more about
[SpeechBrain](https://speechbrain.github.io).

### Inference on GPU
To perform inference on the GPU, add  `run_opts={"device":"cuda"}`  when calling the `from_hparams` method.

### Training
The model was trained with SpeechBrain (aa018540).
To train it from scratch follows these steps:
1. Clone SpeechBrain:
```bash
git clone https://github.com/speechbrain/speechbrain/
```
2. Install it:
```
cd speechbrain
pip install -r requirements.txt
pip install -e .
```

3. Run Training:
```
cd  recipes/IEMOCAP/emotion_recognition
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder=your_data_folder
```

You can find our training results (models, logs, etc) [here](https://drive.google.com/drive/folders/15dKQetLuAhSyg4sNOtbSDnuxFdEeU4zQ?usp=sharing).
