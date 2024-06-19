import os  # 导入os模块，用于操作系统功能，如获取进程ID
from datetime import datetime  # 导入datetime模块，用于获取当前时间
from inspect import currentframe, getframeinfo  # 导入inspect模块，用于获取当前函数的调用栈信息

class MeowLogger(object):  # 定义MeowLogger类
    def __init__(self):  # 初始化方法
        self.logf = None  # 初始化日志文件属性为None

    def __del__(self):  # 析构方法，当对象被销毁时调用
        if self.logf is not None:  # 如果日志文件已经被打开
            self.logf.close()  # 关闭日志文件

    def __header(self, pid):  # 私有方法，生成日志头部信息
        now = datetime.now()  # 获取当前时间
        frameInfo = getframeinfo(currentframe().f_back.f_back)  # 获取调用日志方法的上一级函数的信息
        # 根据是否需要进程ID生成不同的头部格式
        if pid:
            return "[\033[90m{}|\033[0m{}:{}|{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno, os.getpid())
        return "[\033[90m{}|\033[0m{}:{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno)

    def setLogFile(self, filename):  # 设置日志文件的方法
        if self.logf is not None:  # 如果日志文件已经打开，则先关闭
            self.logf.close()
        self.logf = open(filename, "w")  # 打开指定的日志文件，准备写入

    def log(self, content, muted=False):  # 记录日志的方法
        if muted:  # 如果muted为True，则不记录日志
            return
        if self.logf is not None:  # 如果已经设置了日志文件
            self.logf.write(content + "\n")  # 写入日志内容，并添加换行符
            self.logf.flush()  # 刷新日志文件缓冲区，确保内容写入
            return
        print(content)  # 如果没有设置日志文件，则直接打印内容

    # 下面是一系列记录不同颜色日志的方法
    def inf(self, line, pid=False, muted=False):  # 普通日志
        self.log(self.__header(pid) + line, muted)

    def grey(self, line, pid=False, muted=False):  # 灰色日志
        self.log("{}\033[90m{}\033[0m".format(self.__header(pid), line), muted)

    def red(self, line, pid=False, muted=False):  # 红色日志
        self.log("{}\033[91m{}\033[0m".format(self.__header(pid), line), muted)

    def green(self, line, pid=False, muted=False):  # 绿色日志
        self.log("{}\033[92m{}\033[0m".format(self.__header(pid), line), muted)

    def yellow(self, line, pid=False, muted=False):  # 黄色日志
        self.log("{}\033[93m{}\033[0m".format(self.__header(pid), line), muted)

    def blue(self, line, pid=False, muted=False):  # 蓝色日志
        self.log("{}\033[94m{}\033[0m".format(self.__header(pid), line), muted)

    def pink(self, line, pid=False, muted=False):  # 粉色日志
        self.log("{}\033[95m{}\033[0m".format(self.__header(pid), line), muted)

    def cyan(self, line, pid=False, muted=False):  # 青色日志
        self.log("{}\033[96m{}\033[0m".format(self.__header(pid), line), muted)

# 实例化MeowLogger类，并命名为log，作为全局日志记录工具
log = MeowLogger()

