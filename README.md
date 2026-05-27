## 1.UV

https://docs.astral.sh/uv/getting-started/installation/

上面是uv的网址,整个目录用uv做包管理; 安装uv之后只需执行:
uv sync 

就能同步所有依赖环境;



## 2. cartpole.xml

根据小车的一些参数(从pdf文件里面扒出来的)建模的环境,但是并不一定准确;

导入xml文件即可使用该仿真模型,见下面.



## 3. lqr_control.py

根据小车驱动程序(control.c)中的LQR算法copy过来的,理论上算法在mujoco环境中应该能平衡,但是并不能平衡;不过可以看出该算法在尝试平衡小车.

uv run  python3 lqr_control.py  查看仿真

不能平衡的原因分析:

1.

小车里驱动程序还有采样,编码,仿真,PID控制等...还没看明白,最后的输出PWM和力矩之间关系不清楚, 这里使用了一个缩放参数来调试的关系(效果不好),在line 103 step函数的scalar部分,实际调用在line 163的 run里面;

2.

Mujoco建模不准确(不知道咋改)



## 4. cart_pole_env.py

将Mujoco仿真模型与Gymnasium的接口对齐封装,使用gymnasium的框架搭建的一个强化学习框架,后续算法只需要调用该Env框架就可以,

### 重点:

step()方法中,line 133-158部分是reward的设计,这部分是重点,目前设计的不好导致训练效果非常一般.整个Env也可以酌情修改.可以重点分析这一部分如何设计;



## 5.ppo_MultiCore.py

简单的尝试用4.中的Env训练一个RL model出来,执行:
uv run python3 ppo_MultiCore.py

会在指定dir下存储model和对应的log文件



## 6.predict.py

用5.中train出的model 渲染看效果

uv run python3 predict.py



## 7.Summary

目前效果一般,可以尝试:

1.修改cart_pole_env.py中的rewards设计

2.选择更好的算法

3.修改mujoco模型(xml文件)

4.阅读control.c小车代码

的方式提升?  



看到time.sleep()都是用来放慢渲染速度的,嫌慢可以改.


## New
主要修改如下：
1.	电机力矩映射到仿真真实范围：-30，+30，之前的映射只使用了一小部分电机力矩，没有最大化电机输出能力
2.	奖励函数设计具体见代码：train_ppo_v13.py
3.	初始扰动减小为1°附近，让模型更快收敛，后续可以在此基础上将扰动扩大
4.	将仿真中的摩擦系数从1增大到5，减小滑动摩擦
目前最优（注意，奖励曲线和最大步长曲线并未收敛，增大训练步数仍可以提升模型性能，本次模型训练耗时22min，总计300万步）的模型存储在./proj/mujoco_env/saved_models/PPO_CartPole_20260527_0925_3000000 目录下，可以使用predict.py程序运行可视化仿真
