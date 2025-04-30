# Learning to Predict Advertisement Expansion Moments in Short-Form Video Platforms

Dataset and code of our paper **Learning to Predict Advertisement Expansion Moments in Short-Form Video Platforms** (ICMR 2025 full paper).

Authors: [Wenxuan Hou](https://hou9612.github.io/), [Kaibing Yang](https://keibingyang.github.io/), and [Di Hu](https://dtaoo.github.io/).



### WCAE Dataset

Due to the data privacy policy of Tencent, we provide extracted features for each video. Specifically, we use AST to extract audio features, use Swin Transformer and Video Swin Transformer to extract visual features.

Extracted features (~6.4G): [Google Drive](https://drive.google.com/file/d/1pZ6CoH3i-9qZAkvx2EzoXpZhOJ2RasN9/view?usp=drive_link), [Quark Drive](https://pan.quark.cn/s/5074247b0ca3) (password: 1Emm).

Label files (~826K): [Google Drive](https://drive.google.com/file/d/1Y0Gz24Oz3VEXrHLRmXAOHDR7zDj_FjLC/view?usp=drive_link), [Quark Drive](https://pan.quark.cn/s/5e71482683e7) (password: bRuD).



### Requirements

```
python==3.9.16
torch==2.0.0
torchvision=0.15.1
pandas=2.0.1
numpy==1.23.5
```



### Running

```sh
sh run.sh
```

You can find the description of each arg in main.py. 



### Publication(s)

Coming Soon.



### Acknowledgement

We deeply thank Hao Lin, Chong Peng, and Gong Chen in the WeChat Advertisement Department of the Corporate Development Group of Tencent for their support in data collection and processing.

The source code referenced [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20).



### License

This project is released under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).