# Anomaly_Detection
Assuming that the features are distributed across a Gaussian Distribution

I calculated the optimal threshold value by finding out F1 score for all threshold values.

F1 score = 2*(precision * recall)/(precision+recall).
Epsilon(i.e. threshold) = 8.99E-05



I have implemented an anomaly detection algorithm to detect anomalous behavior in server computers. The features measure the through-
put (mb/s) and latency (ms) of response of each server. I have used a Gaussian Model to find the anomalous points in the dataset.
Then fit the Gaussian distribution by calculating mean and variance of each feature. 
------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
