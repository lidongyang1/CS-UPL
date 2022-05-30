# User-Preference-Learning-based-Proactive-Edge-Caching-for-D2D-Assisted-Wireless-Networks
## Abstract
This work investigates proactive edge caching for device-to-device (D2D) assisted wireless networks, where user equipment (UE) can be selected as caching nodes to assist content delivery for reducing the content transmission latency. Doing so, there are two challenging problems: 1) How to precisely learn the user's preference to cache the proper contents on each UE; 2) How to replace the contents cached on UEs when there are new popular contents continuously emerging. To address these, a user preference learning-based proactive edge caching (UPL-PEC) strategy is proposed in this work. In the strategy, we first propose a novel context and social-aware user preference learning method to precisely predict user's dynamic preferences by jointly exploiting the context correlation among different contents, the influence of social relationships and the time-sequential patterns of user's content requests. Specifically, we utilize the bidirectional long short term memory networks to capture the time-sequential patterns of user's content request. And, the graph convolutional networks are adopted to capture the high-order similarity representation among different contents from the constructed content graph. To learn the social influence representation, an attention mechanism is developed to generate the social influence weights to users with different social relationship. Based on the user preference learning, 
a learning-based proactive edge caching architecture is proposed to continuously caching the popular contents on UEs by integrating the offline caching
content placement and the online caching content replacement policy. Real-world trace-based simulation results show that the proposed UPL-PEC strategy
outperforms the compared existing caching strategies at about 2.5\%-45.3\% in terms of the average content transmission latency.

## Dataset
We uploaded the processed dataset to：https://pan.baidu.com/s/1i9R6PJDiEhxxhTgAr8fdLQ 
Extraction Code：g97g
## Training
```
python CS-GCN-LSTM.py #To train the model in single user scenario

python CS-GCN-LSTM_AllUsers.py #To train the model in multiple user scenarios

```
## Predict

```
python CS-GCN-LSTM_predict.py #To predict the results in single user scenario

python CS-GCN-LSTM_AllUsers_predict.py #To predict the results in multiple user scenarios
```

## Results


##

