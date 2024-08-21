# 简介
将MARL环境加入
计划采用QMix算法进行训练

主程序是main.py

# 环境
环境用之前搭建的环境足以, 如果新搭建了一个环境，可以运行
```bash
pip install -r pymarl/requirements.txt
```
进行快速安装

# 环境的观测
由于环境的观测需要是numpy array数组
但是我们的观测又是可变长度的观测
所以作出如下改动
##
由于视野范围是15
所以将蛇头附近30 * 30的方格作为自己的观测的一部分（900维）
然后是10个bean的位置（10 * 2 = 20维）
board_width ,board_height （2维）
再加上三个人蛇头的位置，一共是3 * 2 = 6维
一共是928维

除此之外还要加入上次的动作(4)
自己的id(one hot 3维)
928 + 7 = 935