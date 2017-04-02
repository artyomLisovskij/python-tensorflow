# Recognition of tuberculosis on fluorographic images using deep neural networks PYTHON 3 with Tensorflow+tflearn

## Tech data
You need python3.4 and pip. Project w/o GPU support(CPU only).

pip3 install todo:
> sudo pip3 install tflearn
> sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp34-cp34m-linux_x86_64.whl
> sudo pip3 install pydicom

## About this version
- there is no image preprocessing. i put raw dicom to tensorflow and trying to get results.

## TODO
* image preprocessing(masks?)

## Вопросы:
- какие из изображений с патологией??
- нужно больше данных для обучения

## В аннотацию: 
### какую нейронную сеть выбрали и почему:
- AlexNet, GoogleNet, Inception-Resnet-v2(это из презы булгакова)
- https://deeplearning4j.org/compare-dl4j-torch7-pylearn#theano лучше взять отсюда, не знаю насколько булгаков чист

Это отчет булгакова:
http://docplayer.ru/35920310-Raspoznavanie-tuberkuleza-na-flyuorograficheskih-snimkah-s-pomoshchyu-glubokih-neyronnyh-setey.html
