# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:03:23 2020
包图网视频爬取
@author: luohenyueji
"""

import os
import queue
import threading
import time
import requests
from bs4 import BeautifulSoup
from lxml import etree
import re


# ----- 时间装饰器，打印运行时间
def usetime(func):
    def inner(*args, **kwargs):
        time_start = time.time()
        # 装饰的函数在此运行
        func(*args, **kwargs)
        time_run = time.time() - time_start
        print(func.__name__ + '用时 %.2f 秒' % time_run)

    return inner


class Baotu(object):
    """
    负责爬虫存储
    TODO:
        目标网络有反爬虫机制，多线程下导致有些目标下载失败
        1.解决多线程下网络错误：增加retry机制
        2.异步进行
    注意点：
    requests.text与requests.context区别
    """

    # ----- 初始化
    def __init__(self, url='https://ibaotu.com/shipin/', thread=1, max_page=250, useragent=None, getOriginFile=False):
        """
        :param url: 链接名，如果是自己搜索关键词下载，如搜索中国则链接为https://ibaotu.com/s-shipin/zhongguo.html
        :param thread: 线程数
        :param max_page: 下载最大页数
        :param useragent: 自定义浏览器headers
        :param getOriginFile: 是否获取原始视频
        """
        
        # url为包图网视频链接
        self.url = url
        # 线程数
        self.thread = thread
        # 最大页数
        self.page = max_page
        self.useragent = useragent
        self.header = self._get_header()
        # 请求队列
        self.que = queue.Queue()
        # 失败队列
        self.fail = queue.Queue()
        # 检测当前共有多少分页若用户输入大于当前页面分页则使用当前分页
        page = self._get_maxpage()
        if self.page > page:
            self.page = page
        # 是否下载原始文件
        self.getOriginFile = getOriginFile
        super(Baotu, self).__init__()

    # ----- 设置浏览器类型
    def _get_header(self):
        if not isinstance(self.useragent, str):
            self.useragent = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
            return {'User-Agent': self.useragent}

    # ----- 检测当前共有多少分页
    def _get_maxpage(self):
        req = requests.get(self.url, headers=self.header, timeout=10, verify=False).text
        html = etree.HTML(req)
        # 获得总页数
        # xpath找到class为pagelist的div标签下第8个a标签下的text文本
        pageNum = int(html.xpath("//div[@class='pagelist']/a[8]/text()")[0])
        return pageNum

    # ----- 生产者模型，获取请求列表
    @usetime
    def request(self):
        for i in range(1, self.page + 1):
            try:
                print(self.url)
                # 页数后面会变化自行设置
                req = requests.get(self.url + '7-0-0-0-0-{}.html'.format(i), headers=self.header, timeout=10,
                                   verify=False)
                print('正在爬取第%d页的数据' % i)
                if req.status_code == 200:
                    bs = BeautifulSoup(req.text)
                    # bs找到所有video以及class为scrollLoading的img标签
                    # 第一个findall找到视频地址，第二个findall找到视频名称
                    for _, n in zip(bs.find_all('video', src=True), bs.find_all('img', {'class': 'scrollLoading'})):
                        # 将每个视频组成字典形式放入队列中，{url:视频url,name:视频的名字)}
                        self.que.put({'url': 'http:' + _['src'], 'name': n['alt']})
            except Exception as e:
                print(e)
                pass
        # 计算队列的长度及存储多少视频字典
        print('共有{}条视频需要下载！'.format(self.que.qsize()))

    # ----- 消费者模型，进行下载
    # 默认下载路径为当前文件夹下
    @usetime
    def download(self, path=os.getcwd()):
        # 判断如果队列不为空进行下载
        while not self.que.empty():
            # 从队列中取出视频字典
            data = self.que.get()
            # 如果要下载原始视频
            if self.getOriginFile:
                url = data['url']
                data['url'] = re.sub("mp4_10s.mp4", "mp4", url)
            try:
                req = requests.get(url=data['url'], headers=self.header, verify=False)
                if req.status_code == 200:
                    print('-' * 10, data['url'], '-' * 10)
                    # 如果用户设置path不存在，则创建该path
                    if not os.path.exists(path):
                        os.mkdir(path.strip().rstrip('\\'))
                    # 保存数据
                    # os.path.splitext(data['url'])[-1]为文件后缀名
                    with open(os.path.join(path, data['name'] + os.path.splitext(data['url'])[-1]), 'wb') as f:
                        f.write(req.content)
                else:
                    # 如果请求失败，等待2秒重新下载，感觉没用
                    time.sleep(2)
                    req2 = requests.get(url=data['url'], headers=self.header, verify=False)
                    if req2.status_code == 200:
                        print('+' * 10, data['url'], '+' * 10)
                        with open(os.path.join(path, data['name'] + os.path.splitext(data['url'])[-1]), 'wb') as f:
                            f.write(req.content)
                    else:
                        # 将失败的字典存入fail队列中
                        self.fail.put(data)
                        print(data['name'] + '\t' + '下载失败！')
            except Exception as e:
                print(e)
                continue

    # ----- 控制线程，进行工作
    def run(self):
        # request线程，进行生产者任务
        t1 = threading.Thread(target=self.request)
        # 线程启动
        t1.start()
        # 等待其他线程结束，再结束线程
        t1.join()
        thread_list = []

        # 创建多个download线程
        for i in range(self.thread):
            t = threading.Thread(target=self.download, args=('./save',))
            thread_list.append(t)
        # 开启线程
        for t in thread_list:
            t.start()
        # 子线程全部加入，主线程等所有子线程运行完毕
        for t in thread_list:
            t.join()
        return self.fail.queue

if __name__ == '__main__':
    failQueue = Baotu(max_page=1, thread=4, getOriginFile=True).run()
    print("-" * 50)
    print("失败条数：{}条".format(len(failQueue)))
