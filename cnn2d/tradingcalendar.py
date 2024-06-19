import os  # 导入os模块，用于操作文件和目录路径
import bisect  # 导入bisect模块，用于在有序列表中进行高效的查找

class Calendar(object):  # 定义Calendar类，用于处理交易日历
    def __init__(self):  # 初始化方法
        # 获取日历文件的路径，并读取所有交易日，存储为排序后的列表和集合
        calendarFile = os.path.join(os.path.dirname(__file__), "resources/calendar")
        with open(calendarFile) as f:
            tokens = f.read().splitlines()
            self.tradingDays = sorted([int(x) for x in tokens])  # 交易日列表
            self.tradingDaySet = set(self.tradingDays)  # 交易日集合，便于快速查找

    def isTradingDay(self, date):  # 判断给定日期是否为交易日
        # 确保date是整数，然后检查它是否在交易日集合中
        if not isinstance(date, int):
            date = int(date)
        return date in self.tradingDaySet

    def toTradingDay(self, date):  # 将给定日期调整到最近的交易日
        # 使用bisect_left找到插入点，返回最接近的交易日
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_left(self.tradingDays, date)
        return self.tradingDays[index]

    def next(self, date):  # 获取给定日期的下一天，如果该日是交易日则返回次日
        # 使用bisect_right找到插入点，返回右侧的交易日
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_right(self.tradingDays, date)
        if index >= len(self.tradingDays):
            return None  # 如果已经是最后续的交易日，则返回None
        return self.tradingDays[index]

    def prev(self, date):  # 获取给定日期的前一天，如果该日是交易日则返回前一天的交易日
        # 使用bisect_left找到插入点，返回左侧的交易日
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            return None  # 如果是列表中的第一个日期或之前，则返回None
        return self.tradingDays[index - 1]

    def shift(self, date, n):  # 将给定日期向后（n为正）或向前（n为负）移动n个交易日
        if not isinstance(date, int):
            date = int(date)
        # 确保n是整数
        if not isinstance(n, int):
            return None
        index = bisect.bisect_left(self.tradingDays, date)
        # 确保index在有效范围内
        if index == 0:
            return None
        return self.tradingDays[index + n]

    def prevn(self, date, n):  # 获取给定日期前的n个交易日
        if not isinstance(date, int):
            date = int(date)
        # 确保n是正整数
        if not isinstance(n, int) or n < 1:
            return None
        index = bisect.bisect_left(self.tradingDays, date)
        # 确保index在有效范围内
        if index == 0:
            return None
        # 返回从index-n到index的日期列表，如果不足n个则返回所有剩余的
        return self.tradingDays[max(index - n, 0) : index]

    def nextn(self, date, n):  # 获取给定日期后的n个交易日
        if not isinstance(date, int):
            date = int(date)
        # 确保n是正整数
        if not isinstance(n, int) or n < 1:
            return None
        index = bisect.bisect_right(self.tradingDays, date)
        # 确保index在有效范围内
        if index >= len(self.tradingDays):
            return None
        # 返回从index开始的n个交易日，如果不足n个则返回所有剩余的
        return self.tradingDays[index : min(index + n, len(self.tradingDays))]

    def range(self, startDate, endDate):  # 获取两个日期之间的所有交易日
        if not isinstance(startDate, int):
            startDate = int(startDate)
        if not isinstance(endDate, int):
            endDate = int(endDate)
        # 确保开始日期早于结束日期
        if startDate > endDate:
            return None
        # 使用bisect找到起始和结束日期的插入点，返回范围内的交易日列表
        startIndex = bisect.bisect_left(self.tradingDays, startDate)
        endIndex = bisect.bisect_right(self.tradingDays, endDate)
        return self.tradingDays[startIndex : endIndex]

