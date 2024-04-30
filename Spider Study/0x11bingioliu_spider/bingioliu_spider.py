# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 05:58:50 2020
bingioliu网站必应每日高清壁纸爬取
@author: luohenyueji
"""
import requests
from lxml import etree
from fake_useragent import UserAgent
import os
import time


# ----- 获得网页
def get_one_page(page_start):
    print('当前爬取第{}页'.format(i))
    # 这里下载排行榜的图片
    # url = 'https://bing.ioliu.cn/?p={}'.format(page_start)
    url = 'https://bing.ioliu.cn/ranking?p={}'.format(page_start)
    headers = {'User-Agent': UserAgent().random}
    res = requests.get(url, headers=headers, verify=False)
    if res.status_code == 200:
        return res.text
    else:
        return None


# ----- 解析网页与图片下载
def parse_one_page(html, saveDir='./save'):
    # 生成目录
    os.makedirs(saveDir, exist_ok=True)

    parseHtml = etree.HTML(html)
    picList = parseHtml.xpath('//img/@src')
    # 遍历图片链接
    for pic in picList:
        try:
            # http://h1.ioliu.cn/bing/SantoriniAerial_ZH-CN9367767863_640x480.jpg?imageslim
            # 更换为1920x1080分辨率图片
            picUrl = pic.split('_640')[0] + '_1920x1080.jpg'
            # 图片名字
            picName = pic.split('bing/')[-1].split('_')[0] + '.jpg'
            headers = {'User-Agent': UserAgent().random}
            picRes = requests.get(picUrl, headers=headers)
            # 保存图片
            with open(os.path.join(saveDir, picName), 'wb') as f:
                f.write(picRes.content)

        except Exception as e:
            print(pic, e)
            continue


if __name__ == '__main__':
    for i in range(1, 12):
        # 分页爬取
        html = get_one_page(i)
        parse_one_page(html, saveDir='./save')

        # 由于是个人网站，建议分段延时爬取
        time.sleep(1)
