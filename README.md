# Text classification scripts

## unsloth_classification.ipynb

This modified Unsloth notebook trains LLaMa-3 on any text classification dataset, where the input is a csv with columns "text" and "label".

### Added features:

- Trims the classification head to contain only the "Yes" and "No" tokens, which saves 1 GB of VRAM, allows you to train the head without massive memory usage, and makes the start of the training session more stable.
- Only the last token in the sequence contributes to the loss, the model doesn't waste its capacity by trying to predict the input
- includes "group_by_length = True" which speeds up training significantly for unbalanced sequence lengths
- Efficiently evaluates the accuracy on the validation set using batched inference

## bert_classification.ipynb

This notebook can be used to train any bert model on any text classification dataset (same format as above). The notebook also includes "group_by_length = True" which not commonly found in bert-training notebooks (they usually tokenize everything ahead of time with a lot of wasteful padding).