# Source Code

## Requirements
pytorch

python3

## Command
data_path: the path that contains data files (train.json, dev.json, test.json, image.pkl, dict.json)

out_path: the path for log file and checkpoints
### Train & Eval
```
python train.py -mode train -data_path xxx/xxx/xxx -out_path xxx/xxx/xxx/
```
### Test
```
python train.py -mode ranking -data_path xxx/xxx/xxx -restore xxx/xxx/xxx/best_checkpoint.pt
```
### Inference
```
python train.py -mode generate -data_path xxx/xxx/xxx -restore xxx/xxx/xxx/best_checkpoint.pt
```

## Notice
(1) There are overlapped comments of same videos across the training and test set in the livebot official released processed data. So we give a new split for our experiments in 
```
VideoIC/src/livebot_new_split
```

(2) In our paper, the max_len of a comment is set to 15 (including <BOS> and <EOS> tokens). But we found that it is 20 for both livebot and videoic dataset to get the reported results in our implementation.

(3) For ranking test, we followed the implementation of Livebot to get the candidate set. However, Their actual implementation and the statement in the paper are not align (refer to [response to livebot](https://arxiv.org/abs/2006.03022)). Our statement in Videoic inherited some of these mistakes and also had some other misrepresentations. However, it should be noted that the experimental results of all datasets in our paper were obtained under the same test setting, so these problems didn't affect the fairness of the comparison. We correct the statement here:

Our candidate set contains 100 candidates, at most 5 for ground-truth comments, others are improper comments. We first pool all comments in the training set excluding the ground-truth comments for the target time stamp together as the candidate comments set Z. Then we sample improper comments in three ways from Z:
- Popular comments: 20 comments with the highest frequency (appears more than 100 times) in the candidate comments set Z.
- Plausible comments: 20 comments most similar (cosine distance between the TF-IDF vectors) to the context comments in the candidate comments set Z.
- Random comments: Add comments randomly picked from the candidate comments set Z till 100.

