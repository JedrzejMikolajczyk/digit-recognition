# MNIST Playground

This project allows [PyTorch](https://github.com/pytorch) users working with the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to test how their models predict on their **custom input**.

|![](https://github.com/JedrzejMikolajczyk/digit-recognition/blob/main/GIF.gif)|
|-|
## Installation
```
pip install -r requirements.txt
```

## How to use
### To open the program:
```
python main.py
```

### To add your model:

To use your own model and weights please click 'Add model' and then specify your model's name, file, and weights' file.  

|![](https://github.com/JedrzejMikolajczyk/digit-recognition/blob/main/picture1.png)|
|-|

Alternatively, you can add them manually to ```settings.json```  

### Switching between models:
|![](https://github.com/JedrzejMikolajczyk/digit-recognition/blob/main/GIF1.gif)|
|-|

### To retrain weights:
```
python predictor.py [OPTIONAL ARGS]
```


