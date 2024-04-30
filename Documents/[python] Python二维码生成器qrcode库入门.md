# Python二维码生成器qrcode库入门

qrcode是二维码生成的Python开源库，官方地址为[python-qrcode](https://github.com/lincolnloop/python-qrcode)

# 1 简介

qrcode依赖于pillow，安装代码如下：

> pip install qrcode[pil]

**什么是二维码QRCode？**

快速响应码Quick Response Code（QRCode）是一种二维象形码，具有快速的可读性和较大的存储容量。 该码由在白色背景上以方形图案排列的黑色模块组成（可以更改颜色）。 编码的信息可以由任何类型的数据组成（例如，二进制、字母数字或汉字符号）。二维码能存储大量的数据，将所有数据存储为方形网格中的一系列像素。二维码详细的细节和原理见[二维码的生成细节和原理](https://blog.csdn.net/wangguchao/article/details/85328655)。


# 2 用法
## 2.1 基础使用

### 2.1.1 命令行使用

从命令行，使用已安装的qr脚本：


```python
! qr "hello world!" > test1.png
```

然后我们可以在当前路径获得一个名为test1.png的二维码文件，图像宽高为290。图片显示代码如下：


```python
from PIL import Image
from IPython.display import display
 
img = Image.open('test1.png', 'r')
print("img size is {}".format(img.size))

# 显示图片
display(img)
```

    img size is (290, 290)



![png](image/output_5_1.png)


### 2.1.2 Python接口

在 Python 中，使用make快捷功能，也可以输出二维码图像，代码如下：


```python
import qrcode
# 构建二维码
data = 'hello world!'
img = qrcode.make(data)
# 显示图片格式，为qrcode.image.pil.PilImage
print(type(img))
# 保存图片 
img.save("test2.png")
```

    <class 'qrcode.image.pil.PilImage'>


### 2.1.3 二维码解析

如果想查看生成的二维码信息，可以用手机扫描二维码，或者使用[草料二维码解析器](https://cli.im/deqr)在线解析图片。解析结果如下图所示：


![](https://gitee.com/luohenyueji/article_picture_warehouse/raw/master/Python-Study-Notes/qrcode/image/result.png)

## 2.2 高级使用
### 2.2.1 二维码自定义
我们还可以通过在之前使用该QRCode函数创建的qr对象中添加一些属性来自定义QR 码的设计和结构。基本参数如下：

+ version：一个1 到40之间的整数，用于控制 QR 码的大小（最小的版本1是一个21x21矩阵）。默认为None，表示代码自动确认该参数。
+ error_correction：用于二维码的纠错。qrcode 包中提供了以下四个常量：
	1. ERROR_CORRECT_L 大约可以纠正 7% 或更少的错误。
	2. ERROR_CORRECT_M （默认）大约 15% 或更少的错误可以被纠正。
	3. ERROR_CORRECT_Q 大约 25% 或更少的错误可以被纠正。
	4. ERROR_CORRECT_H 大约可以纠正 30% 或更少的错误。
+ box_size：控制二维码的每个“盒子”有多少像素，默认为10。
+ border：控制边框应该有多少个框厚（默认为 4，这是根据规范的最小值）。


```python
import qrcode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
data = "hello world!"
qr.add_data(data)
qr.make(fit=True)

# fill_color和back_color分别控制前景颜色和背景颜色，支持输入RGB色，注意颜色更改可能会导致二维码扫描识别失败
img = qr.make_image(fill_color=( 213 , 143 , 1 ), back_color="lightblue")
display(img)
```


![png](image/output_10_0.png)


### 2.2.2 二维码输出
我们还可以将二维码可以导出为SVG图片。


```python
import qrcode
import qrcode.image.svg
method = 'fragment'
if method == 'basic':
    # Simple factory, just a set of rects.
    # 简单模式
    factory = qrcode.image.svg.SvgImage
elif method == 'fragment':
    # Fragment factory (also just a set of rects)
    # 碎片模式
    factory = qrcode.image.svg.SvgFragmentImage
else:
    # Combined path factory, fixes white space that may occur when zooming
    # 组合模式，修复缩放时可能出现的空白
    factory = qrcode.image.svg.SvgPathImage

img = qrcode.make('hello world!', image_factory=factory)

# 保存图片 
img.save("test3.svg")
```

### 2.2.3 二维码图像样式

要将样式应用于QRCode，请使用StyledPilImage。这需要一个可选的module_drawers参数来控制二维码的形状，一个可选的color_mask参数来改变二维码的颜色，还有一个可选的embeded_image_path参数来嵌入图像。这些二维码并不能保证对所有的二维码识别器都有效，所以做一些实验并将纠错error_correction设置为高（尤其是嵌入图像时）。


python-qrcode提供的二维码的形状列表如下：

![](https://gitee.com/luohenyueji/article_picture_warehouse/raw/master/Python-Study-Notes/qrcode/image/module_drawers.png)

python-qrcode提供的二维码的颜色列表如下：


![](https://gitee.com/luohenyueji/article_picture_warehouse/raw/master/Python-Study-Notes/qrcode/image/color_masks.png)


具体使用代码如下：


```python
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer,SquareModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask,SquareGradiantColorMask

# 纠错设置为高
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
# 如果想扫描二维码后跳转到网页，需要添加https://
qr.add_data('https://www.baidu.com')

# 修改二维码形状
img_1 = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer())
# 修改二维码颜色
img_2 = qr.make_image(image_factory=StyledPilImage, color_mask=SquareGradiantColorMask())
# 嵌入图像
img_3 = qr.make_image(image_factory=StyledPilImage, embeded_image_path="lena.jpg")
# 嵌入图像
img_4 = qr.make_image(image_factory=StyledPilImage, module_drawer=SquareModuleDrawer(), color_mask=RadialGradiantColorMask(), embeded_image_path="lena.jpg")
```


```python
img_1
```




![png](image/output_15_0.png)




```python
img_2
```




![png](image/output_16_0.png)




```python
img_3
```




![png](image/output_17_0.png)




```python
img_4
```




![png](image/output_18_0.png)



# 3 参考

+ [python-qrcode](https://github.com/lincolnloop/python-qrcode)
+ [草料二维码解析器](https://cli.im/deqr)
+ [二维码的生成细节和原理](https://blog.csdn.net/wangguchao/article/details/85328655)
