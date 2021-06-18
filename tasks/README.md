## Tasks
This folder contains data for tasks related to the VideoIC dataset


---


### Live Comments Generation

#### Task Introduction
The goal of live video comments generation is to generate human-like comments at target time stamp T of the given video based on surrounding context.

#### Dataset

Dataset division in paper [VideoIC: A Video Interactive Comments Dataset and Multimodal Multitask Learning for Comments Generation](https://dl.acm.org/doi/10.1145/3394171.3413890) is under
```
/VideoIC/task/comments_generation/division/
```
We split the data into training set, development set, and test set.
For the dict in the split file, attribute and explanation is below:

Attribute | Meaning
--- | ---
aid | the only identification of a video
cid | the only identification of a comments on the bilibili website
title  | the title of a video shown on the bilibili website
duration | the duration of a video (in seconds)
comment_number  | the number of comments we got of a video
class | the category of a video

#### Processed data
We  provide processed data in our paper under the folder
```
/VideoIC/task/comments_generation/processed_data
```
, including dictionary, processed training set , development set and testing set data.


---



### Multi-Label Video Classification

#### Task Introduction
This work explores the multi-label video classification task assisted by danmaku. Multi-label video classification can associate multiple tags to a video from different aspects, which can benefit video understanding tasks such as video recommendation. We collects tags for VideoIC videos and builds a hierarchical label structure for the first time on danmaku video data. 

#### Tags
In paper [弹幕信息协助下的视频多标签分类](http://www.jsjkx.com/CN/10.11896/jsjkx.200800198), we collect tags for each video.


The original tags are organized according to the categories of VideoIC: 
```
VideoIC/task/multilabel_video_classification/tags/ori_tags
```
After manually filtering out infrequent labels, we got a clean version of video tags: 
```
VideoIC/task/multilabel_video_classification/tags/filtered_video_tags.json
```
The key is the **aid** of each video

We provide the hierarchical structure of tags. In the dict file below, there are 3 levels from coarse to fine:
```
VideoIC/task/multilabel_video_classification/tags/hierarch_structure.json
```

We also process the video tags into hierarchical version in:
```
VideoIC/task/multilabel_video_classification/tags/hierarch_filtered_video_tags.json
```
The key is the **aid** of each video. Attribute and explanation is below:

Attribute | Meaning
--- | ---
category | the coarse-grianed label
subfield | mid-grained label
fine_grain  | the fine-grained label

#### Data Split
Dataset division in paper [弹幕信息协助下的视频多标签分类](http://www.jsjkx.com/CN/10.11896/jsjkx.200800198) is under:
```
VideoIC/task/multilabel_video_classification/division
```
We split the data into training set, development set, and test set.
For the dict in the split file, the key is the **aid** of each video. Attribute and explanation is below:

Attribute | Meaning
--- | ---
category | the coarse-grianed label
subfield | mid-grained label
fine_grain  | the fine-grained label
