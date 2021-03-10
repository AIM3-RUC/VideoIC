# VideoIC Dataset

This folder contains all raw data of VideoIC dataset, including videos along with related information and live comments. \
As VideoIC is collected according to six categories, we organize raw data according to their categories.

### Video Information
Information about all videos in the VideoIC dataset is in the file
```
/VideoIC/data/info.json
```
The key is the **aid** of each video, the only identification of a video on the bilibili website. You can access the video by the link
```
https://www.bilibili.com/video/av[aid]
```
for example, the video with aid 10582565 can be accessed by the link
```
https://www.bilibili.com/video/av10582565
```


Attribute and explanation of items in the dict is below:

Attribute | Meaning
--- | ---
cid | the only identification of the comments file
title  | the title of a video shown on the bilibili website
duration | the duration of a video (in seconds)
total_danmaku  | the number of comments we got of a video
v_class | the category of a video
v_key | the keyword we search for collecting the video

**Cid** is the only identification of the live comments of each video. You can access to the latest comments by adding the prefix for the link
```
http://comment.bilibili.com/[cid].xml
```
For example, the latest comments of video with aid 10582565 and cid 18518844 can be accessed by link
```
http://comment.bilibili.com/18518844.xml
``` 



### Videos
All videos of VideoIC dataset is under the folder
```
/VideoIC/data/videos/
```
named by their aid, the only identification of a video on the bilibili website. You can access the video by the link
```
https://www.bilibili.com/video/av[aid]"
```
for example, the video with aid 10582565 can be accessed by the link
```
https://www.bilibili.com/video/av10582565
```
All the videos copyrights belong to the website www.bilibili.com. Please use the videos and comments for research only.




### Live video comments

Comment files for each video is under the folder 
```
 /VideoIC/data/comments/
```
named by the aid of a video.

The comments json file contains:

Key | value
---|---
time(s) | a list of comments at this time stamp




