import win32gui, win32ui, win32con
from PIL import Image
from pyminitouch import MNTDevice
'''
`pyminitouch` 是一个 **Python 封装库**，用于在电脑上通过 **ADB** 控制安卓设备的 **触摸/滑动/多点触控**，底层通常依赖开源的 `minitouch` 工具（把触摸事件注入到 Android 的输入系统中）。它常用于：

- **自动化测试/脚本**：点击、长按、滑动、拖拽等手势
- **手游/应用自动操作**：比纯 `adb shell input tap/swipe` 更稳定、支持更高频率与多点
- **配合截图/识别**：常与 `adb screencap`、`minicap`、OpenCV 等一起用来做“看图点按”的自动化

典型能力（取决于具体封装实现）：
- 连接设备（通过 ADB）
- 发送 touch down / move / up
- 控制触摸点 ID（实现多指操作）
- 调整事件发送频率、坐标映射等

如果你想，我可以根据你当前项目（ResnetGPT）里是否已有相关依赖（如 `pyminitouch/minitouch/adbutils`）帮你定位它被用在哪里、具体怎么用。你希望我在仓库里搜一下吗？
'''
import sys, os
import time


class MyMNTDevice(MNTDevice):
    def __init__(self,ID):
        MNTDevice.__init__(self,ID)


    def 发送(self,内容):
        self.connection.send(内容)

def 取图(窗口名称):
    # 获取后台窗口的句柄，注意后台窗口不能最小化
    hWnd = win32gui.FindWindow(0,窗口名称)  # 窗口的类名可以用Visual Studio的SPY++工具获取
    # 获取句柄窗口的大小信息
    left, top, right, bot = win32gui.GetWindowRect(hWnd)
    width = right - left
    height = bot - top
    # 返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    hWndDC = win32gui.GetWindowDC(hWnd)
    # 创建设备描述表
    mfcDC = win32ui.CreateDCFromHandle(hWndDC)
    # 创建内存设备描述表
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建位图对象准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 为bitmap开辟存储空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    # 将截图保存到saveBitMap中
    saveDC.SelectObject(saveBitMap)
    # 保存bitmap到内存设备描述表
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)


    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    ###生成图像
    im_PIL = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr,'raw','BGRX')
    #im_PIL= Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr)
    #im_PIL =Image.frombytes('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr)
    box = (8,31,968,511)
    im2 = im_PIL.crop(box)
    #im2.save('./dd2d.jpg')
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hWnd, hWndDC)
    return im2


