# NLP_Course - Proposed Method

This folder contains the codes for the proposed method section. There are three python files that each of them is compiled independently. The CRF.py contain the code for applying [CRF.py](https://github.com/soroush-ziaeinejad/NLP_Course/blob/c9cfc1c8f54a81f039c18c551975b440fc593b56/ProposedMethod_ziaeines/src/CRF.py) to our dataset. [RNNLSTM.py](https://github.com/soroush-ziaeinejad/NLP_Course/blob/c9cfc1c8f54a81f039c18c551975b440fc593b56/ProposedMethod_ziaeines/src/RNNLSTM.py) contains the main models (RNN, GRU, and Bidirectional LSTM). 


Table below shows the weighted F1 score for each model that we used to reach the Bidirectional LSTM as the best model.

|      **Model**     | **Training Weighted F1** | **Testing Weighted F1** |
|:------------------:|:------------------------:|:-----------------------:|
|         CRF        |           0.35           |           0.34          |
|      SimpleRNN     |           0.63           |           0.59          |
|         GRU        |           0.96           |           0.61          |
| Bidirectional LSTM |         **0.99**         |         **0.69**        |

The loss and accuracy plots are provided below for SimpleRNN, GRU, and Bidirectional LSTM.

![SimpleRNN](https://github.com/soroush-ziaeinejad/NLP_Course/blob/5fd9faf59be06a32c5750bf9b89c3fee9046b5bc/ProposedMethod_ziaeines/imgs/simpleRNN%20-%20Copy.png)


![GRUreal2](https://github.com/soroush-ziaeinejad/NLP_Course/blob/5fd9faf59be06a32c5750bf9b89c3fee9046b5bc/ProposedMethod_ziaeines/imgs/GRUreal2%20-%20Copy.png)

![lstm](https://github.com/soroush-ziaeinejad/NLP_Course/blob/5fd9faf59be06a32c5750bf9b89c3fee9046b5bc/ProposedMethod_ziaeines/imgs/lstm%20-%20Copy.png)


requirements:
```
pip uninstall scikit-learn
pip install scikit-learn==0.23.2
pip install sklearn-crfsuite
pip install eli5
```


To run each model, you should:

*    **CRF**:

```python CRF.py```

*    **SimpleRNN**:

Change the model_ variable in RNNLSTM file to **'simpleRNN'**

```python RNNLSTM.py```

*    **GRU**:

Change the model_ variable in RNNLSTM file to **'GRU'**

```python RNNLSTM.py```

*    **Bidirectional LSTM**:

Change the model_ variable in RNNLSTM file to **'LSTM'**

```python RNNLSTM.py```
