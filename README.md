# Voice-Interaction-Control-on-Unitree-G1-Robot-to-realze-FALCON-
This project is an integration of Microphone on G1, LLM and FALCON
Step 1
根据https://github.com/LeCAR-Lab/FALCON/tree/main 以及 https://github.com/LeCAR-Lab/FALCON/tree/main/sim2real （主要）中的README.md文档安装配置好环境（检验标准是sim2sim,sim2real均可以正常运行），下载代码（FALCON作为例子在此出现，也可以是其他基于宇树G1机器人的开源仓库）
这一步应该注意
- sim2sim过程中中只允许键盘控制，sim2real允许键盘与XBOX控制，但是在直接从FALCON下载下来的代码中，键盘与XBOX指令属于有交集但互不包含，因此我作了一些补充（黄色高亮部分），以下是对照表
- 建议详细阅读sim2real文件夹中的三个python文件（base_policy.py,loco_manip/loco_manip.py/dec_loco/dec_loco.py）,理清每个变量的含义以及维度
Step 2
- 运行宇树文档中的音频历程，请参考https://support.unitree.com/home/zh/G1_developer/quick_development注意修改文档末尾的历程名。此步成功的标准是G1能够发出声音“你好，我是宇树科技的机器人，历程启动成功”及其英文，并且在绿灯亮起后对着麦克风说话可以输出转化后的文本。
<img width="900" height="953" alt="image" src="https://github.com/user-attachments/assets/2b3c03dd-54ea-4c1f-bf5d-7ffc6df755a3" />

Step 3
- 实现指令的分类（ChatGPT）以及麦克风例程与ChatGPT的通信（LCM）。
- 安装LCM （C++与python版本都要安装）
sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install lcm
- 在unitree_sdk2/example/g1/audio文件夹下新建.lcm文件（需要学习LCM通信方式）以及llm.api.py文件，详见https://github.com/lunow0715/Voice-Interaction-Control-on-Unitree-G1-Robot-to-realize-FALCON-/tree/master/add_to_(unitree_sdk2)
- 同时运行麦克风历程C++代码和大语言模型python代码
<img width="1257" height="809" alt="image" src="https://github.com/user-attachments/assets/2de392fc-0927-47c7-a685-023d1455164d" />
Step 4
- 将以上两个模块整合到FALCON中，需要修改FALCON代码，详情可见https://github.com/lunow0715/Voice-Interaction-Control-on-Unitree-G1-Robot-to-realze-FALCON-/tree/master中loco_manip部分，添加了一部分代码，头文件等等，文件目录也可以参考
- 运行时需要开两个终端，一个运行麦克风例程，一个运行FALCON sim2real部分的deploy部分


Experience and Lessons
- 关于G1
  1. 左臂下方是电池以及开机键，短按之后长按开机，更换电池时需要对准接口，注意方向（红色标签向上），详情可参照https://support.unitree.com/home/zh/G1_developer/quick_start中的“开机流程”
  2. 使用G1上的麦克风需要切换到唤醒模式（XBOX按L1+L2切换，共三个模式 wake up mode,key mode,close interaction）
  3. 连接G1与显示器的接口松动，能否连接上要看运气，不过本项目可以不用连接显示器
  4. sim2real时要保证机器人完全放下，下降到可以独立站立的高度
  5. FALCON策略开始运行时（由放松切换到站立姿态）动作可能比较激烈，请做好心理准备并适当远离
- 关于FALCON中的代码
  1. sim2real/config/g1/g1_29dof_falcon.yaml  sim2sim时不需要修改，sim2real需要将接口替换为G1的网口
  INTERFACE: "lo" # in simulation, lo0 for mac, lo for linux
  #INTERFACE: "enx00e04c682a9d" # 根据ping192.168.123.164得到的网口而定
  并将
  USE_JOYSTICK: 0 # Simulate Unitree WirelessController using a gamepad (0: disable, 1: enable)
  JOYSTICK_TYPE: "xbox" # support "xbox" and "switch" gamepad layout
  JOYSTICK_DEVICE: 0 # Joystick number
  中的0改为1
  2. 手部旋转问题（能够录入，接收并映射，应该是FALCON中代码问题）（似乎在sim2sim中就没有反应）
    这是由于FALCON禁用了手腕动作（虽然他有控制手腕动作的指令），需要修改loco_manip.py中的代码以解除这个禁用
  <img width="923" height="1280" alt="image" src="https://github.com/user-attachments/assets/a5e20507-03ae-4717-bf8e-d3251e62569c" />
  <img width="932" height="591" alt="image" src="https://github.com/user-attachments/assets/e531977a-6c31-4a0e-b34c-9536e9aefc4d" />
  <img width="1280" height="369" alt="image" src="https://github.com/user-attachments/assets/5fa13863-b849-41f6-9faf-bec814ba58d4" />
- 关于sim2real
  请务必先在Mujoco环境中进行sim2sim,确保安全（有能够熟练操控G1的使用者陪同）的情况下再sim2real

Todo
1.探索策略结束时能否较为“温柔的”恢复，目前方案是先用“速度恢复为0”，再用“站立/踏步状态切换”切换到站立姿态。直接结束策略双臂会突然泄力。可以阅读宇树文档修改loco_manip.py中的结束策略代码使得结束策略时执行与XBOX按L2+B(阻尼模式)时相同的函数


2.解决“踏步时声音过大麦克风难以收录声音”的问题，（外接麦克风？修改宇树的g1_audio_client_example.cpp文件以支持？） 

3.处理较为复杂的指令（“向前走5m”），可以考虑借助ChatGPT分解为简单指令

主要参考网页：https://support.unitree.com/home/zh/G1_developer
            https://github.com/LeCAR-Lab/FALCON/tree/main/sim2real
