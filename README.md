# A comparison between a RNN, Random Forestand Na ̈ıve Bayes in sentiment analysis
### Jille van der Togt
### March 2020
## 1. Abstract
Sentiment analysis is one of the most important aspects of natural language processing (NLP). Its usage varies a lot, from customer service to personal assistants. The aim of sentiment analysis is to understand what emotion the user is trying to express. In this research we compared three algorithms and their performance in categorizing movie reviews as either a positive or negative sentiment. The goal was to find the best model and in which areas they excel. We created a recurrent neural network, a naïve Bayes classifier and a random forest. All three models were tested on accuracy, training time, sample time and storage space. The accuracy of all three models was quite similar but they differed a lot in the other metrics. The conclusion is that each model has a specific application area which is determined by how the user plans on using the model. 
## 2. Introdcution
### 2.1 Background
Sentiment analysis is becoming increasingly important in the use of text categorization. Sentiment analysis tries to assess what emotion the user is displaying. Sentiment analysis is necessary because there are vast amounts of data needed to be analyzed, and this exceeds human capacity. For big companies, who get tens of thousands of reviews and feedback pieces per month, it is important that these get interpreted, labeled and used for improvement of processes. There are several reasons for this. For instance, (Lipsman, 2007) concluded that between 73% and 87% of online shoppers base their purchase on reviews. Also (Hochreiter, 1998) concluded that consumers are willing to pay more than 20% extra on products with a five-star review than a four-star review. If these companies can correctly classify feedback pieces, learn from it and process these lessons, they can better satisfy customers. This is just one example of why sentiment analysis is important but there are many others: personal assistants like Siri, translation services like Google translate or products like chat bots and customer support. This paper aims to develop three different models to predict the sentiment of a text sample and then compares these three models in terms of accuracy, time taken and storage space. The results might be used by companies in deciding which model for sentiment analysis to choose from: a recurrent neural network (RNN), a Naïve Bayes classifier or a random forest.

### 2.2 Questions
This paper aims to answer the question which of the three models will perform the best in sentiment analysis on the IMDB dataset of movie reviews (Maas et al., 2011).  The questions which need to be answered are:
* Which model performs best in terms of accuracy? *The  RNN  is  expected  to  outperform  the  other  classifiers  in  terms  of  ac-curacy.   This  is  because  this  model  is  versatile;  it  can  change  and  adapt to new instances well.  This is not the case for the random forest because it overfits extremely and the Bayes will lack as well because it is a Naïve model.*
* Which model trains the quickest? *The  random  forest  will  probably  perform  the  best  in  this  area  because  it overfits  extremely  on  the  training  data  and  thus  simply  follows  structure found there.  The RNN will have  to fine-tune millions  of weights and the Naïve  Bayes  must  calculate  the  probabilities  for  all  unique  words,  which both take longer to train.*
* Which model is most usable in a real-time application (what is the sample time)? *The Naïve Bayes will presumably outperform the other models here because it only needs to look up the words in a list and multiply their probabilities. The RNN and the random forest will be slower because they must run the sample through the complete network before a classification can be made.*
* Which model is the most memory efficient? *The  random  forest  is  expected  to  be  most  memory  efficient  because  it doesn’t  have  to  save  a  lot  of  data,  just  a  few  constraints,  while  the  RNN has  to  save  millions  of  weights  and  the Naïve  Bayes  a  very  large  list  ofprobabilities.*

## 3 Method
### 3.1 Software
The models are all made in python 3.62.  The RNN is trained with the keras library.  The random forest is trained with the sklearn libraray.  The Naïve Bayes is our own implementation.  All code can be found on Github. Other software used are libraries like numpy and utils from NLTK. The rest of the smaller libraries can be found in the code.

### 3.2 Data
There is a total of 50k labeled examples in the dataset.  25k positive and 25k negative reviews.  These are split into 12.5k training and 12.5k testing samples. A list of all unique words in the vocabulary and links to the actual reviews are provided.  There are also 50k unlabeled documents for unsupervised learning, these are not used in this research.

### 3.3 Recurrent Neural Network
Recurrent  neural  networks  have  become  state-of-the-art  models  and  replaced almost all forms of back-off models (the state-of-the-art models up to that point)due to better performance (Mikolov, Karafi ́at, Burget,ˇCernocky, & Khudanpur,2010).  In this paper the RNN uses a long short-term memory cell (LSTM cell). This  cell  enables  the  usage  of  past  experiences  (the  history  of  the  input)  to classify the current sentence.  This cell solves the vanishing gradient problem(Hochreiter, 1998).The  input  of  the  model  is  a  sentence  but  because  the  length  of  sentences differs a lot and our network wants the same input length every time, we limit the  length  of  the  sentence  to n=500  characters.   Sentences  which  are  shorter than n get padded to fit.  Because the network cannot take in words as input we convert all words to integers by searching for the index of the word in the given imdb.vocab file. The  first  layer  of  the  model  is  an  embedding  layer.   It  uses  a  100k  word vocab, 32 embedding size and n as input length.  The second layer in the model is a LSTM cell of 100 units.  The last part of the model is a dense layer, which is the same as a normal densely connected neural net.  The output space is 1unit, so either positive or negative.  This layer uses the sigmoid function as an activation function. The training of the model is done via a binary cross-entropy loss function ,with the adam optimizer (Kingma & Ba, 2014) and it saves the accuracy of the model for later use.  The model is run for three epochs with a batch validation size of 64 samples.

### 3.4 Naïve Bayes
The  Naïve Bayes  uses  Bayes’  theorem  in  combination  with  an  independence assumption between features, which is why the classifier is called Naïve (most features  do  depend  on  each  other).   It  uses  word  frequencies  as  features  and calculates  their  parameters  as  the  maximum  likelihood  estimation  (MLE)  of these frequencies (Rish et al., 2001).  The MLE is: 

![\phi_f^{(y)} = \frac{count_{YF}(y, f)}{count_Y(y)}](https://render.githubusercontent.com/render/math?math=%5Cphi_f%5E%7B(y)%7D%20%3D%20%5Cfrac%7Bcount_%7BYF%7D(y%2C%20f)%7D%7Bcount_Y(y)%7D)

The  probability  of  a  word  is  the  count  of  that  word  in  a  context  divided  by the count of all words in that context.  To deal with unseen words we use add-α smoothing  (Goldwater,  2019)  (the  hyper  parameter  was  found  to  be  best at α=  1).   We  run  this  MLE  over  all  training  examples  for  both  positive and negative and end up with two large dictionaries (word:  probability).  To predict the class of a new sample〈f1.....fn〉we solve the following maximization problem:

![y^* = \underset{y}{argmax} \; P_{Y\vert F^n_1} (y\vert \langle  f_1.....f_n  \rangle)](https://render.githubusercontent.com/render/math?math=y%5E*%20%3D%20%5Cunderset%7By%7D%7Bargmax%7D%20%5C%3B%20P_%7BY%5Cvert%20F%5En_1%7D%20(y%5Cvert%20%5Clangle%20%20f_1.....f_n%20%20%5Crangle))

![= \underset{y}{argmax} \; \frac{P_Y(y) P_{F^n_1\vert Y} (\langle  f_1.....f_n  \rangle \vert y)}{P_{F^n_1}(\langle  f_1.....f_n  \rangle)}](https://render.githubusercontent.com/render/math?math=%3D%20%5Cunderset%7By%7D%7Bargmax%7D%20%5C%3B%20%5Cfrac%7BP_Y(y)%20P_%7BF%5En_1%5Cvert%20Y%7D%20(%5Clangle%20%20f_1.....f_n%20%20%5Crangle%20%5Cvert%20y)%7D%7BP_%7BF%5En_1%7D(%5Clangle%20%20f_1.....f_n%20%20%5Crangle)%7D)

![= \underset{y}{argmax} \; P_Y(y) P_{F^n_1\vert Y} (\langle  f_1.....f_n  \rangle \vert y)](https://render.githubusercontent.com/render/math?math=%3D%20%5Cunderset%7By%7D%7Bargmax%7D%20%5C%3B%20P_Y(y)%20P_%7BF%5En_1%5Cvert%20Y%7D%20(%5Clangle%20%20f_1.....f_n%20%20%5Crangle%20%5Cvert%20y))

![= \underset{y}{argmax} \; \prod^n_{i=1} P_{F\vert Y} (f_i \vert y)](https://render.githubusercontent.com/render/math?math=%3D%20%5Cunderset%7By%7D%7Bargmax%7D%20%5C%3B%20%5Cprod%5En_%7Bi%3D1%7D%20P_%7BF%5Cvert%20Y%7D%20(f_i%20%5Cvert%20y))

![= \underset{y}{argmax} \; \sum_{i=1}^n log \; \phi^{(y)}_{f_{i}}](https://render.githubusercontent.com/render/math?math=%3D%20%5Cunderset%7By%7D%7Bargmax%7D%20%5C%3B%20%5Csum_%7Bi%3D1%7D%5En%20log%20%5C%3B%20%5Cphi%5E%7B(y)%7D_%7Bf_%7Bi%7D%7D)

### 3.5 Random Forest
Random forest classifiers are often used in classification and can be made by constructing multiple decision trees (Safavian & Landgrebe, 1991) and running a sample over all these trees to output the mode of the classes of the individual trees.   Random  forests  are  well  known  to  extremely  overfit  on  their  training data  (Segal,  2004).   Different  trees  use  different  features  to  split  the  samples over and thus classify differently.  After enough splitting the samples end up in one category which is their label.

![Alt text](iris-1.jpg?raw=true "Title")

### 3.6 Testing
All three models are evaluated the same way.  The test samples get run through the model.  Based on what the model predicts and the true label, we determine the total accuracy of the model.

![accuracy = \frac{\textit{\# good classifications}}{\textit{\# total classifications}}](https://render.githubusercontent.com/render/math?math=accuracy%20%3D%20%5Cfrac%7B%5Ctextit%7B%5C%23%20good%20classifications%7D%7D%7B%5Ctextit%7B%5C%23%20total%20classifications%7D%7D)

To measure the training time, we time how long the program takes to train and test all 50 thousand samples.  To measure sample time, we run one whole test set through the model and record that time.  Memory usage gets determined by the size of the model and not the size of the data because the data is comparable amongst all models.

## 4 Results
The metrics of all three models can be found below.

Table 1: Statistics of all three models
Model name | Accuracy | Train time (sec) | Sample time (sec) | Memory usage (MB)
-----------|----------|------------------|-------------------|------------------
RNN | 0.853 | 1253 | 60 | 39.0
Naïve Bayes | 0.836 | 6649 | 1.49 | 4.6
Random Forest | 0.844 | 215 | 219 | 110.6

## 5 Discussion
### 5.1 Accuracy
In  the  second  column  in  table  1  the  accuracies  of  the  models  are  displayed. All three accuracies are comparable.  The RNN performs the best.  It performs 1.7% better than the worst model, the Naïve Bayes.  The random forest is almost exactly in between the other two.  These results co-inside with the hypotheses. Because  all  accuracies  are  this  close  to  each  other  we  can  assume  all  models will classify new samples with the about same accuracy.  This means we should compare them on other areas to see which one performs the best overall.

### 5.2 Train time
In  training  time,  the  random  forest  out-performs  the  other  two  significantly. This model finished 25k samples in 215 seconds.  The RNN performed the task in 1253  seconds  (20  minutes),  a  600%  increase  from  the  random  forest.   The Naïve Bayes  performed  the  worst.   It  finished  training  in  6649  seconds  (two hours).  These results were expected in the hypotheses.

### 5.3 Sample time
The Naïve Bayes outperformed the other models by a large margin.  It finished 25k samples less than 1.5 second.  The RNN finished in exactly one minute and the random forest took 219 seconds (3.5 minutes).  These results are also in line with the hypotheses.

### 5.4 Memory Space
The Naïve Bayes performed the best in this category. The model required 4.6 MB of storage space.  The RNN took almost 40 MB. The random forest required well  over  100  MB.  These  results  are  not  in  line  with  the  hypotheses.   The random forest needs to save a lot of data because it needs to make thousands of comparisons while the Naïve Bayes only needs to store a large list with numbers and the RNN a few million parameters (just floats).

### 5.5 Future work
There  are  still  improvements  to  be  made  for  both  the  RNN  and  the  random forest, the Naïve Bayes has been optimized.  A lot of combinations of hyper parameters were tried, but none of these had a positive effect on the performance.
  For  the  RNN  it  is  possible  to  add  more  (different)  layers  in  the  network. This could possibly lead to an increase of a few percent in terms of accuracy because the model is able to represent another dimension of information in the text.  Adding more layers will worsen the time and space metrics because it hast o fine-tune and store more parameters than before.  Also adding layers does not mean an improvement in accuracy since larger models can overfit to data faster  and  thus  worsen  the  test  set  accuracy.   This  was  not  tried  yet  due  to time-constraints in training these networks.
  The  random  forest  also  has  room  for  improvement,  mostly  in  the  hyperparameters.  Sklearn offers more than 10 hyper parameters to change and this could possibly increase the performance of the model.  For example, a change in minimal node impurity could lead to a better generalized model because it is smaller and thus less overfitted.  The limiting of the amount of leaf nodes can also lead to a smaller model.  Or maybe a lower maximum on the amount of features it looks over per split can increase how well the model generalizes.
  Another improvement could be adding extra features to the dataset.  The IMDB  dataset  contains  a  file  with  URL’s  to  all  reviews  and  from  there  it  is possible to extract extra features.  For instance, the number of users that found the review helpful or the number of stars given to the movie.  These features will not help the Naïve Bayes, since that model only uses word counts, but the other two could gain performance by considering these features.

## 6 Conclusion
We presented different models and their usage in sentiment analysis. Our results show  that  each  of  the  three  models  has  almost  the  same  accuracy  and  their own advantages and disadvantages.  The model that suits you best will depend entirely on the usage.  If a quick scanning is necessary every time a user enters a review (in large companies for example), the Naïve Bayes should be the used model.  But if the model is gradually changing over time, the random forest will be a better fit.  If your application is anywhere in between, the RNN fits best.

## References
Goldwater, S.  (2019).  Anlp lecture 6 n-gram models and smoothing. School ofInformatics.

Hochreiter, S. (1998). The vanishing gradient problem during learning recurrentneural nets and problem solutions. International Journal of Uncertainty,Fuzziness and Knowledge-Based Systems,6(02), 107–116.

Kingma, D. P., & Ba, J.  (2014). Adam:  A method for stochastic optimization.arXiv preprint arXiv:1412.6980.

Lipsman, A. (2007). Online consumer-generated reviews have significant impacton offline purchase behavior. comscore.Inc. Industry Analysis, 2–28.

Maas,  A.  L.,  Daly,  R.  E.,  Pham,  P.  T.,  Huang,  D.,  Ng,  A.  Y.,  &  Potts,  C.(2011,  June).    Learning  word  vectors  for  sentiment  analysis.    InPro-ceedings  of  the  49th  annual  meeting  of  the  association  for  computationallinguistics:  Human  language  technologies(pp. 142–150).  Portland, Ore-gon,  USA:  Association  for  Computational  Linguistics.   Retrieved  fromhttp://www.aclweb.org/anthology/P11-1015

Mikolov, T., Karafi ́at, M., Burget, L.,ˇCernocky, J., & Khudanpur, S.  (2010).Recurrent neural network based language model. InEleventh annual con-ference of the international speech communication association.

Rish, I., et al.  (2001).  An empirical study of the naive bayes classifier.  InIjcai2001 workshop on empirical methods in artificial intelligence(Vol. 3, pp.41–46).

Safavian,  S. R., & Landgrebe,  D.  (1991).  A survey of decision tree classifiermethodology.IEEE transactions on systems, man, and cybernetics,21(3),660–674.

Segal, M. R. (2004). Machine learning benchmarks and random forest regression.UCSF: Center for Bioinformatics and Molecular Biostatistics.  Retrievedfromhttps://escholarship.org/uc/item/35x3v9t4
