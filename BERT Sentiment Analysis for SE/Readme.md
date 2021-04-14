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
 
