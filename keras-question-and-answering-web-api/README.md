Question answering system developed using seq2seq and memory network model in Keras. In this project we worked with two different encoding for the input of the encoder (one hot encoding and GloVe word2vec encoding) while merging the paragraph context and the question as the input of the encoder. 
The implemented models include:
Seq2seq.py : One-hot encoding input that is paragraph_context + ' Q ' + question
Seq2seq_v2.py : One-hot encoding input that is add(paragraph_context, RepeatVector(LSTM(question)))
Seq2seq_glove.py : GloVe encoding input that is paragraph_context + ' Q ' + question
Seq2seq_v2_glove.py : GloVe encoding input that is add(paragraph_context, RepeatVector(LSTM(question)))

The trained models are included in the demo folder in the project . The training was done using the Full-Abstract/ BioASQ-train-factoid-7b-snippet data set on with 200 epochs and batch size of 64.


The figure below compare the training accuracy and validation accuracy of various models using the script squad_compare_models:
 

In seq2seq-qa-v2 one hot encoding seq2seq used but it is different from seq2seq-qa as in seq2seq-qa-v2 the paragraph context and the question are added after the LSTM + RepeatVector layer. While in seq2seq-qa-glove data was trained on word-level (GloVe word2vec encoding) with input = paragraph_context + ' Q ' + question) and in Seq2seq-qa-v2-glove the paragraph context and the question are added after the LSTM + RepeatVector layer. To summarize, the RepeatVector is used as an adapter to fit the fixed-sized output of the encoder to the differing length and input expected by the decoder.
Model
F1-Score
seq2seq-qa
0.7999999
seq2seq-v2-qa
0.8166667
seq2seq-glove-qa
0.7499999
seq2seq-glove-v2-qa
0.7999999
