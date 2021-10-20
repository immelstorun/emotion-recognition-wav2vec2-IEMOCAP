---
language: "en"
thumbnail:
tags:
- speechbrain
- classification
- Emotion
- Recognition
- wav2vec2
- pytorch
license: "apache-2.0"
datasets:
- iemocap
metrics:
- Accuracy
---

<iframe src="https://ghbtns.com/github-btn.html?user=speechbrain&repo=speechbrain&type=star&count=true&size=large&v=2" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
<br/><br/>

# Emotion Recognition with wav2vec2 base on IEMOCAP

This repository provides all the necessary tools to perform emotion recognition with a fine-tuned wav2vec2 (base) model using SpeechBrain. 
It is trained on IEMOCAP training data.


For a better experience, we encourage you to learn more about
[SpeechBrain](https://speechbrain.github.io). The model performance on IEMOCAP test set is:

| Release | Accuracy(%) | 
|:-------------:|:--------------:|
| 19-10-21 | 78.7 (Avg: 75.3) | 


## Pipeline description

This system is composed of an wav2vec2 model. It is a combination of convolutional and residual blocks. The embeddings are extracted using attentive statistical pooling. The system is trained with Additive Margin Softmax Loss.  Speaker Verification is performed using cosine distance between speaker embeddings.

## Install SpeechBrain

First of all, please install the **development** version of SpeechBrain with the following command:

```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

Please notice that we encourage you to read our tutorials and learn more about
[SpeechBrain](https://speechbrain.github.io).

### Perform Emotion recognition

```python
import torchaudio
from speechbrain.pretrained.interfaces import EncoderWav2vecClassifier
classifier = EncoderWav2vecClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
signal, fs =torchaudio.load('/workspace/emotion-recognition-wav2vec2/anger.wav')
prediction = classifier.classify_batch(sig)
```
 The prediction tensor will contain a tuple of (embedding, id_class, label_name).

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

### Limitations
The SpeechBrain team does not provide any warranty on the performance achieved by this model when used on other datasets.

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/
