# asr.py
import whisper
import numpy as np


class ASR:
    """
    语音识别模型类
    """

    def __init__(self, modelType: str = "small", prompt: str = ""):
        """
        :param modelType: whisper模型类型
        :param prompt: 提示词
        """
        # 模型默认使用cuda运行，没gpu跑模型很慢。
        # 使用device="cpu"即可改为cpu运行
        self.model = whisper.load_model(modelType, device="cuda")
        # prompt作用就是提示模型输出指定类型的文字
        # 这里使用简体中文就是告诉模型尽可能输出简体中文的识别结果
        self.prompt = "简体中文" + prompt

    def predict(self, audio: np.ndarray) -> str:
        """
        语音识别
        :param audio: 输入的numpy音频数组
        :return: 输出识别的字符串结果
        """

        # prompt在whisper中用法是作为transformer模型交叉注意力模块的初始值。transformer为自回归模型，会逐个生成识别文字，
        # 如果输入的语音为空，initial_prompt的设置可能会导致语音识别输出结果为initial_prompt
        result = self.model.transcribe(audio.astype(np.float32), initial_prompt=self.prompt)
        return result["text"]