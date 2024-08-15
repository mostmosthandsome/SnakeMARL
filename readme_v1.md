# Snake3v3

### 示例代码：

~~~python
import numpy as np
import pdb
from env.snakes import SnakeEatBeans
from submissions.red import policy as red_policy
from submissions.blue import policy as blue_policy

env = SnakeEatBeans()
obs = env.reset(render=True)

action_dim = env.get_action_dim()
num_player = len(env.players)

while not env.is_terminal():
    
    action_red = red_policy(obs[:3])
    action_blue = blue_policy(obs[3:])

    all_actions = action_red + action_blue
    next_obs, reward, terminal, info = env.step(all_actions)
    state = env.get_global_state()
    
print(env.check_win())
~~~

### 环境说明

* 通过SnakeEatBeans创建环境，在创建环境过程中读入config文件，config.json为游戏内容的配置项，包括地图大小、豆子数量等。该文件在代码调试初期可以修改为简单的问题测试代码正确性，在正式开发的时候不建议修改。
* env.reset()方法初始化游戏并返回**红蓝双方所有智能体**的观测数据，其传入的render参数为bool变量：如果传入True则有pygame实时渲染对局情况；如果传入False则只有状态机推演而没有图形，由于没有图形，推演速度快得多。
* env.is_terminal()判断游戏是否结束（是否达到最大的timestep），在env.step()的返回值terminal中也可以用来判断是否结束。
* 通过red_policy和blue_policy函数根据传入的obs来决策行为，其中由于obs返回了6个智能体的观测，因此红色方使用obs[:3]，蓝色方采用obs[3:]。返回的action以one-hot列表的形式存在，可以先输出一下看一下形式。
* env.step输入6个智能体的行为，环境进行推演后得到下时刻的obs，reward，terminal，和info。
* env.get_global_state()函数可以获取全局的状态，该函数在CTDE的假设下训练中可以使用，但是在对局中不可以使用。
* env.check_win()判断一局游戏中6个智能体最后的获胜的智能体，根据获胜的智能体属于哪一方判断赢家。
* obs中加入了视野的限制，每个智能体只能看到距离它曼哈顿距离10格内的对手部分，而队友的部分可以完全看到。所以在render的时候发现智能体不按自己的预想行进可能是视野问题。
* obs每一个维度的具体说明可以参考[text](http://www.jidiai.cn/env_detail?envid=6)

### 提交说明

* 实际推演中将建立submissions文件夹，将各位队伍的提交结果放入其中，队伍需要提交一个网络模型（如果采用了神经网络）以及调用模型的python文件(war.py)。python文件中覆盖policy函数即可，我们将调用该函数进行模拟，policy函数的签名不要修改。

~~~
|submissions
|  |team1
|  |  |war.py
|  |  |model.pth
|  |team2
|  |  |war.py
|  |  |model.pth
...
~~~

* 由于推演需要红蓝双方，因此在自己训练的时候也需要提供双方文件。初期可以采用随机策略或者自博弈的方式训练。后面如果队伍间分别训练出模型也可以互相使用对方的作为对手。
* 代码方面可以先行训练，具体提交方式和每日结果公布情况尚在商议。

**由于代码方面刚改好，所以存在有bug的可能性，如果发现实现上的bug，请联系我们。**

**最后祝大家取得好成绩😁**

