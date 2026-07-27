![](https://user-images.githubusercontent.com/47793918/233812617-beab2e71-57b9-479e-8bff-c3931347ca40.png)

##在安装 sp-master260612 之前，请先切换到 spbig260427-2 分支，随后再安装本分支
本分支从 sunnypilot 的 master 分支 fork 而来, 260427是fork日期；

  1、优化了纵向，将原车纵向和OP视觉纵向融合，弯道减速、高速状态远距离优先选择OP预刹车；

  2、增加了驾驶员接管方向盘大角度转向后，回正方向盘，存在抢方向盘的问题；
  
  3、优化跟车起步响应；
  
  4、优化刹车，允许超过设定速度，优先使用油门减速，再启用EPS刹车；
  
  5、增加高德地图分屏显示，在配置了高德服务密钥和端密钥后，且WIFI联网正常，可以显示高德分屏地图；
  
  6、从bp中获取了网页服务器，手机浏览器可以访问openpilot，配置参数，包括配置高德参数，目的地址设置；
  
  7、横向增加了跟随前方弯道最大曲率，动态改变横向曲率提前量；理论上能过更大弯道；
  
  8、修改了panda的部分横向限制；横向最大步径角度能提高一点；

  9、从bp中获取了盲点监测系统 显示渐变红，更醒目也更美观；
  


## 什么是 sunnypilot？
[sunnypilot](https://github.com/sunnyhaibin/sunnypilot)是comma.ai的openpilot项目的分支，后者是一款开源驾驶辅助系统。sunnypilot为用户提供了独特的驾驶体验，支持300多种汽车品牌与车型，并对驾驶辅助功能的触发行为进行了定制化调整。sunnypilot 尽可能严格地遵守 comma.ai 的安全规范。

## 加入我们的社区论坛
加入官方的 sunnypilot 社区论坛，及时了解所有最新功能，并参与塑造 sunnypilot 的未来！
* https://community.sunnypilot.ai/

##文档
https://docs.sunnypilot.ai/ 是您了解 sunnypilot 各项功能、安装指南及常见问题解答的一站式入口。

##在汽车中的专用设备上运行
首先，请查看这份您需要准备的事项清单[入门指南](https://community.sunnypilot.ai/t/getting-started-using-sunnypilot-in-your-supported-car/251).

##安装
接下来，请参阅 sunnypilot 社区论坛中的安装说明](https://community.sunnypilot.ai/t/read-before-installing-sunnypilot/254)，以及完整的[推荐分支安装列表]

##拉取请求
我们欢迎在GitHub上提交拉取请求和问题。鼓励修复漏洞。

拉取请求应基于最新的`master`分支。

##用户数据

默认情况下，sunnypilot 会将驾驶数据上传至 comma 服务器。您也可以通过[comma connect](https://connect.comma.ai/).

sunnypilot 是一款开源软件。用户如需，可自行选择禁用数据收集功能。

sunnypilot 会记录面向道路的摄像头、CAN 总线、GPS、IMU、磁力计、温度传感器数据，以及车辆碰撞信息和操作系统日志。
面向驾驶员的摄像头和麦克风仅在您于设置中明确选择启用时才会被记录。

通过使用本软件，您理解：使用本软件或其相关服务将产生特定类型的用户数据，这些数据可能会由comma自行决定进行记录和存储。在您同意本协议的同时，您即授予comma一项不可撤销、永久且全球范围内的权利，以用于处理上述数据。

## Licensing

sunnypilot is released under the [MIT License](LICENSE). This repository includes original work as well as significant portions of code derived from [openpilot by comma.ai](https://github.com/commaai/openpilot), which is also released under the MIT license with additional disclaimers.

The original openpilot license notice, including comma.ai’s indemnification and alpha software disclaimer, is reproduced below as required:

> openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.
>
> Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.
>
> **THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
> YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
> NO WARRANTY EXPRESSED OR IMPLIED.**

For full license terms, please see the [`LICENSE`](LICENSE) file.

## 💰 Support sunnypilot
If you find any of the features useful, consider becoming a [sponsor on GitHub](https://github.com/sponsors/sunnyhaibin) to support future feature development and improvements.


By becoming a sponsor, you will gain access to exclusive content, early access to new features, and the opportunity to directly influence the project's development.


<h3>GitHub Sponsor</h3>

<a href="https://github.com/sponsors/sunnyhaibin">
  <img src="https://user-images.githubusercontent.com/47793918/244135584-9800acbd-69fd-4b2b-bec9-e5fa2d85c817.png" alt="Become a Sponsor" width="300" style="max-width: 100%; height: auto;">
</a>
<br>

<h3>PayPal</h3>

<a href="https://paypal.me/sunnyhaibin0850" target="_blank">
<img src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" alt="PayPal this" title="PayPal - The safer, easier way to pay online!" border="0" />
</a>
<br></br>

Your continuous love and support are greatly appreciated! Enjoy 🥰

<span>-</span> Jason, Founder of sunnypilot
