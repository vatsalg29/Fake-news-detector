# Fake-news-detector
Deep Learning model to detect fake news on the LIAR_PLUS Dataset

## Running the code
1. Download the dataset here - [link](https://aclweb.org/anthology/W18-5513)
2. Assign column names to the tsv files
3. The notebook has necessary changes for it to be able to run in Google colab. Kindly remove those changes to run in your personal machine.

## Results
The model achieves 44.12% test accuracy on the 6 way classification task and 73.56% on the binary classification task.

## Architecture
I've used a Bi-LSTM + Conv architecture for this problem. The model takes all the attributes as input, gets their embedding, 
then passes them through 2 BiLSTM layers consisting of 32 and 16 neural units respectively. Then I pass them through a Dense layer, 
reshape them and pass through a convolutional layer followed by a max pool layer. These features are extracted for every attribute 
except the history count. Then these features are combined 2 at a time by :

1. Relation between Statement and Statement type
2. Relation between Statement and Context 
3. Relation between Speaker and Party. 
4. Relation between Party and Speaker’s job. 
5. Relation between Statement type and Context.
6. Relation between Statement and State.
7. Relation between Statement and Party.
8. Relation between State and Party.
9. Relation between Context and Party.
10. Relation between Context and Speaker.
11. Relation between Statement and Justification.
12. Relation between Context and Justification.

After this, all these different combos are combined together, merged with the history, passed through a few dense layers and then the 
final layer of 6 neurons with softmax activation. All other layers have relu activation. The model was trained using Adam optimizer for
both tasks and for around 20 epochs.
