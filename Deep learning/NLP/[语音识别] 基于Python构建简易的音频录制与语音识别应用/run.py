# run.py
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QSize, Qt
import sys
from record import AudioHandle


class Window(QMainWindow):
    """
    界面类
    """

    def __init__(self):
        super().__init__()
        # --- 设置标题
        self.setWindowTitle('语音识别demo')
        # --- 设置窗口尺寸
        # 获取系统桌面尺寸
        desktop = app.desktop()
        # 设置界面初始尺寸
        self.width = int(desktop.screenGeometry().width() * 0.3)
        self.height = int(0.5 * self.width)
        self.resize(self.width, self.height)
        # 设置窗口最小值
        self.minWidth = 300
        self.setMinimumSize(QSize(self.minWidth, int(0.5 * self.minWidth)))

        # --- 创建组件
        self.showBox = QTextEdit()
        self.showBox.setReadOnly(True)
        self.startBtn = QPushButton("开始录音")
        self.stopBtn = QPushButton("停止录音")
        self.stopBtn.setEnabled(False)

        # --- 组件初始化
        self.initUI()

        # --- 初始化音频类
        self.ahl = AudioHandle()
        # 连接用于传递信息的信号
        self.ahl.infoSignal.connect(self.showInfo)
        self.showInfo("<font color='blue'>{}</font>".format("程序已初始化"))

    def initUI(self) -> None:
        """
        界面初始化
        """
        # 设置整体布局
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.showBox)
        # 设置底部水平布局
        blayout = QHBoxLayout()
        blayout.addWidget(self.startBtn)
        blayout.addWidget(self.stopBtn)
        mainLayout.addLayout(blayout)

        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)

        # 设置事件
        self.startBtn.clicked.connect(self.record)
        self.stopBtn.clicked.connect(self.record)

    def record(self) -> None:
        """
        录音控制
        """
        sender = self.sender()
        if sender.text() == "开始录音":
            self.stopBtn.setEnabled(True)
            self.startBtn.setEnabled(False)
            # 开启录音线程
            self.ahl.start()
        elif sender.text() == "停止录音":
            self.stopBtn.setEnabled(False)
            # waitDialog用于等待录音停止
            waitDialog = QProgressDialog("正在停止录音...", None, 0, 0)
            waitDialog.setWindowTitle("请等待")
            waitDialog.setWindowModality(Qt.ApplicationModal)
            waitDialog.setCancelButton(None)
            waitDialog.setRange(0, 0)

            # 设置 Marquee 模式
            waitDialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
            waitDialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
            waitDialog.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
            waitDialog.setWindowFlag(Qt.WindowMinimizeButtonHint, False)
            waitDialog.setWindowFlag(Qt.WindowTitleHint, False)
            # 关闭对话框边框
            waitDialog.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

            # 连接关闭信号，即ahl线程结束则waitDialog关闭
            self.ahl.finished.connect(waitDialog.accept)
            # 结束录音线程
            self.ahl.stop()
            if self.ahl.isRunning():
                # 显示对话框
                waitDialog.exec_()

            # 关闭对话框
            self.ahl.finished.disconnect(waitDialog.accept)
            waitDialog.close()

            self.startBtn.setEnabled(True)

    def showInfo(self, text: str) -> None:
        """
        信息展示函数
        :param text: 输入文字，可支持html
        """
        self.showBox.append(text)
        if not self.ahl.running:
            self.stopBtn.click()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """
        重写退出事件
        :param event: 事件对象
        """
        # 点击停止按钮
        if self.ahl.running:
            self.stopBtn.click()
        del self.ahl
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    # 获取默认图标
    default_icon = app.style().standardIcon(QStyle.SP_MediaVolume)

    # 设置窗口图标为默认图标
    ex.setWindowIcon(default_icon)

    ex.show()
    sys.exit(app.exec_())