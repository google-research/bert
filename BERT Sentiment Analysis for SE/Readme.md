# BERT Sentiment Analysis model for Software Engineering Comments 

We have sentiment analysis model to analyze user reviews , chats , messages , comments , as well product reviews too . Generally the domain of analysis speaks about analysis of the sentiments of movies , people review or any product or service . Right now we dont have as such production models to speak about technical language sentiment analysis .Here in this problem statement , I created a BERT model to do sentiment analysis on the software engineering comments , which can help coders , developers as well site admins to look on the sentiment of the asked questions and here in ground truth lying behind . 

## What is BERT : 

BERT (Bidirectional Encoder Representations from Transformers) is a recent paper published by researchers at Google AI Language. It has caused a stir in the Machine Learning community by presenting state-of-the-art results in a wide variety of NLP tasks, including Question Answering (SQuAD v1.1), Natural Language Inference (MNLI), and others.
BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modelling. This is in contrast to previous efforts which looked at a text sequence either from left to right or combined left-to-right and right-to-left training. The paper’s results show that a language model which is bidirectionally trained can have a deeper sense of language context and flow than single-direction language models. In the paper, the researchers detail a novel technique named Masked LM (MLM) which allows bidirectional training in models in which it was previously impossible.

## Dataset : 
 i collected data from Stack over flow , Git hub , JIRA . I used resources from Kaggle and github to collect the CSV files an the raw text datas . Next i mereged the entire data in a proper structured categorical data format and saved inside the ./data folder . The data is dived into two formats ./data/Train.csv & ./data/Test.csv. The data is having comments from developers and its accompanied by the underneath sentiment.
 
 ## Special Features of the model : 
 
 The special features of this project which speaks about the sake of doing it includes : 
 ```
 a.) It is difficult to analyze the technical keywords and pass it into AI models for sentiment analysis
 b.) If sites like Github , JIRA , Stack overflow have this power of sentiment nalaysis from this type of advanced model called BERT , then they can easily eleimante out spams ,
     plagarism as well can detect which type of content is going quality . Also it will halp a lot in evaluating technlogies and tech stack based on the responses.
 ```
 
 ## How this model will work : 
 
 For training the BERT Model  I am using [K train Library](https://pypi.org/project/ktrain/) which is a a fastai-like interface to Keras, that helps build and train Keras models with less time and coding. ktrain is open-source and available on GitHub [here](https://github.com/amaiya/ktrain/tree/master/ktrain).
 
 To install ktrain, simply type the following:
 ```
 pip install ktrain
 ```
 To begin, let’s import the ktrain and ktrain.text modules:
 ```
 import ktrain
from ktrain import text
```
Load the Data in the BERT model : 
```
train_path="/content/Train.csv"
test_path="/content/Test.csv"
tr_path= pathlib.Path(train_path)
te_path=pathlib.Path(test_path)
if tr_path.exists ():
    print("Train data path set.")
else: 
    raise SystemExit("Train data path does not exist.")
     
if te_path.exists ():
    print("Test data path set.")
else: 
    raise SystemExit("Test data path does not exist.")
    
(x_train, y_train), (x_test, y_test), preproc =  text.texts_from_array(train_df[:,2], train_df[:,1],  x_test=test_df[:,2], y_test=test_df[:,1],maxlen=500, preprocess_mode='bert')
```
Load BERT and wrap it in a Learner object
The first argument to get_learner uses the ktraintext_classifier function to load the pre-trained BERT model with a randomly initialized final Dense layer. The second and third arguments are the training and validation data, respectively. The last argument get_learner is the batch size. We use a small batch size of 6.
```
model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)
learner = ktrain.get_learner(model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test), 
                             batch_size=6)
```
Train the model
To train the model, we use the fit_onecycle method of ktrain which employs a 1cycle learning rate policy that linearly increases the learning rate for the first half of training and then decreases the learning rate for the latter half:
```
learner.autofit(2e-5, early_stopping=5)
```
Plot the learning rate
```
learner.lr_plot()
```
Storing the model
```
model.save("model.h5")
predictor = ktrain.get_predictor(learner.model, preproc)
```

## How to Run the script : 
The steps involved to run the script are as follows : (Specify all your data paths before run)
```
pip install -r requirements.txt
python model.py
```

## Final Conclusion :

The model is performing with near about to 86.7 % accuracy on the testing satge on the test data.
