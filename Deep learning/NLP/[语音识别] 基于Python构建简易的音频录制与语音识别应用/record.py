# record.py
import speech_recognition as sr
from PyQt5.QtCore import QThread, pyqtSignal
import time, os
import numpy as np
from asr import ASR


class AudioHandle(QThread):
    """
    录音控制类
    """

    # 用于展示信息的pyqt信号
    infoSignal = pyqtSignal(str)

    def __init__(self, sampleRate: int = 16000, adjustTime: int = 1, phraseLimitTime: int = 5,
                 saveAudio: bool = False, hotWord: str = ""):
        """
        :param sampleRate: 采样率
        :param adjustTime: 适应环境时长/s
        :param phraseLimitTime: 录音最长时长/s
        :param saveAudio: 是否保存音频
        :param hotWord: 热词数据
        """
        super(AudioHandle, self).__init__()
        self.sampleRate = sampleRate
        self.duration = adjustTime
        self.phraseTime = phraseLimitTime
        # 用于设置运行状态
        self.running = False
        self.rec = sr.Recognizer()
        # 麦克风对象
        self.mic = sr.Microphone(sample_rate=self.sampleRate)
        # 语音识别模型对象
        # hotWord为需要优先识别的热词
        # 输入"秦剑 无憾"表示优先匹配该字符串中的字符
        self.asr = ASR(prompt=hotWord)
        self.saveAudio = saveAudio
        self.savePath = "output"

    def run(self) -> None:
        self.listen()

    def stop(self) -> None:
        self.running = False

    def setInfo(self, text: str, type: str = "info") -> None:
        """
        展示信息
        :param text: 文本
        :param type: 文本类型
        """
        nowTime = time.strftime("%H:%M:%S", time.localtime())
        if type == "info":
            self.infoSignal.emit("<font color='blue'>{} {}</font>".format(nowTime, text))
        elif type == "text":
            self.infoSignal.emit("<font color='green'>{} {}</font>".format(nowTime, text))
        else:
            self.infoSignal.emit("<font color='red'>{} {}</font>".format(nowTime, text))

    def listen(self) -> None:
        """
        语音监听函数
        """
        try:
            with self.mic as source:
                self.setInfo("录音开始")
                self.running = True
                while self.running:
                    # 设备监控
                    audioIndex = self.mic.audio.get_default_input_device_info()['index']
                    workAudio = self.mic.list_working_microphones()
                    if len(workAudio) == 0 or audioIndex not in workAudio:
                        self.setInfo("未检测到有效音频输入设备！！！", type='warning')
                        break
                    self.rec.adjust_for_ambient_noise(source, duration=self.duration)
                    self.setInfo("正在录音")
                    # self.running为否无法立即退出该函数，如果想立即退出则需要重写该函数
                    audio = self.rec.listen(source, phrase_time_limit=self.phraseTime)
                    # 将音频二进制数据转换为numpy类型
                    audionp = self.bytes2np(audio.frame_data)
                    if self.saveAudio:
                        self.saveWav(audio)
                    # 判断音频rms值是否超过经验阈值，如果没超过表明为环境噪声
                    if np.sqrt(np.mean(audionp ** 2)) < 0.02:
                        continue
                    self.setInfo("音频正在识别")
                    # 识别语音
                    result = self.asr.predict(audionp)
                    self.setInfo(f"识别结果为：{result}", "text")
        except Exception as e:
            self.setInfo(e, "warning")
        finally:
            self.setInfo("录音停止")
            self.running = False

    def bytes2np(self, inp: bytes, sampleWidth: int = 2) -> np.ndarray:
        """
        将音频二进制数据转换为numpy类型
        :param inp: 输入音频二进制流
        :param sampleWidth: 音频采样宽度
        :return: 音频numpy数组
        """

        # 使用np.frombuffer函数将字节序列转换为numpy数组
        tmp = np.frombuffer(inp, dtype=np.int16 if sampleWidth == 2 else np.int8)
        # 确保tmp为numpy数组
        tmp = np.asarray(tmp)

        # 获取tmp数组元素的数据类型信息
        i = np.iinfo(tmp.dtype)
        # 计算tmp元素的绝对最大值
        absmax = 2 ** (i.bits - 1)
        # 计算tmp元素的偏移量
        offset = i.min + absmax

        # 将tmp数组元素转换为浮点型，并进行归一化
        array = np.frombuffer((tmp.astype(np.float32) - offset) / absmax, dtype=np.float32)

        # 返回转换后的numpy数组
        return array

    def saveWav(self, audio: sr.AudioData) -> None:
        """
        保存语音结果
        :param audio: AudioData音频对象
        """
        nowTime = time.strftime("%H_%M_%S", time.localtime())
        os.makedirs(self.savePath, exist_ok=True)
        with open("{}/{}.wav".format(self.savePath, nowTime), 'wb') as f:
            f.write(audio.get_wav_data())