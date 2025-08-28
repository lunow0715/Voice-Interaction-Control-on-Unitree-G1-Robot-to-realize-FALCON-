import lcm
import json
from openai import OpenAI
from unitree_asr import AsrTextMessage  # LCM生成的Python代码

# 配置GPT API（替换为你的密钥和URL）
OpenAI.api_key = ""
OpenAI.api_base = ""

# 文本分类到数字的映射（包含所有指令）
CATEGORY_MAPPING = {
    # 基础高度调整
    "基础高度增加0.1米": 301,     # B+上
    "基础高度减少0.1米": 302,     # B+下
    # 腰部俯仰调整
    "腰部俯仰减小0.1弧度": 303,   # Y+上
    "腰部俯仰增加0.1弧度": 304,   # Y+下
    # 左右手末端X轴调整
    "左右手末端X轴增加0.05米": 305, # R1+上
    "左右手末端X轴减少0.05米": 306, # R1+下
    # 左右手末端Y轴调整
    "左末端Y轴增加右末端Y轴减少": 307, # R1+左
    "右末端Y轴增加左末端Y轴减少": 308, # R1+右
    # 左右手末端Z轴调整
    "左右手末端Z轴增加0.05米": 309, # X+上
    "左右手末端Z轴减少0.05米": 310, # X+下
    # 左右手末端旋转
    "左右手末端顺时针转5度": 311,   # X+左
    "左右手末端逆时针转5度": 312,   # X+右
    # 腰部偏航调整
    "腰部偏航减少0.1弧度": 313,    # select+左
    "腰部偏航增加0.1弧度": 314,    # select+右
    # 腰部俯仰微调
    "腰部俯仰减少0.05弧度": 315,   # select+上
    "腰部俯仰增加0.05弧度": 316,   # select+下
    # 控制参数调整
    "站立": 318,                  # R2 #=
    "速度设为0": 319,              # L2 #z
    "启动": 320,                  # start #
    "停止策略": 321,               # B+Y
    "初始状态": 322,               # A+X
    "KP缩放减小0.1": 323,          # Y+左
    "KP缩放增加0.1": 324,          # Y+右
    "KP微调减小0.01": 325,         # A+左
    "KP微调增加0.01": 326,         # A+右
    "KP还原为1": 327,              # A+Y
    # 线速度控制（新增方向键）
    "线速度向前增加": 328,         # 键盘"w"（对应摇杆ly正向）
    "线速度向后增加": 329,         # 键盘"s"（对应摇杆ly负向）
    "线速度向左增加": 330,         # 键盘"a"（对应摇杆lx正向）
    "线速度向右增加": 331,         # 键盘"d"（对应摇杆lx负向）
    "角速度向左增加": 332,         # 键盘"q"（对应摇杆rx正向）
    "角速度向右增加": 333,         # 键盘"e"（对应摇杆rx负向）
    "线速度设为-1.0": 317,         # 键盘"m"
    "其他": 0                     # 未匹配的内容
}

def gpt_classify(text):
    """调用GPT分类文本"""
    try:
        # 实例化OpenAI客户端
        client = OpenAI(
            api_key="",
            base_url=""
        )
        
        response = client.chat.completions.create(  # 注意这里的调用方式
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"请将输入文本分类为以下类别之一：仅返回类别名称，不附加其他内容。{list(CATEGORY_MAPPING.keys())}"},
                {"role": "user", "content": text}
            ]
        )
        category = response.choices[0].message.content.strip()
        # 确保类别与映射中的键完全匹配（区分大小写）
        if category in CATEGORY_MAPPING:
            return category
        else:
            # 尝试不区分大小写匹配
            lower_category = category.lower()
            for key in CATEGORY_MAPPING.keys():
                if key.lower() == lower_category:
                    return key
            return "其他"
    except Exception as e:
        print(f"GPT调用失败: {str(e)}")
        return "其他"


def on_speech_text(channel, data):
    """处理接收到的语音文本"""
    msg = AsrTextMessage.decode(data)
    print(f"收到语音文本: {msg.text}")

    # 调用GPT分类
    category = gpt_classify(msg.text)
    print(f"GPT分类结果: {category}")

    # 映射到数字
    result = CATEGORY_MAPPING.get(category, 0)
    print(f"映射后数字: {result}\n")

if __name__ == "__main__":
    # 初始化LCM并订阅消息
    lc = lcm.LCM()
    lc.subscribe("ASR_TEXT_CHANNEL", on_speech_text)  # 与C++使用相同的通道名
    print("等待接收语音文本...")

    # 循环处理消息
    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        print("程序退出")
