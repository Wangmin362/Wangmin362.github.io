+++
title = 'Windows自定义后台进程并设置为开机启动'
date = 2024-03-22T22:17:28+08:00
draft = false
+++
[TOC]

# 1 背景

自己开发了一个应用程序，想要再`windows`上后台运行，并且能够设置为开机启动。

# 2 目标

可以在`Windows`上配置任意一个可执行文件后台启动，并且设置为开机启动。

# 3 应用程序测试代码

测试代码非常简单，就是再运行之后，每隔三秒钟在可执行文件的同级目录的app-logs文件夹中创建文件。如果程序正常运行，那么我们就可以看到新创建的文件。

编译命令：`go build -o app.exe main.go`

```go
package main

import (
	"fmt"
	"os"
	"time"
)

func main() {

	dir := "app-logs"
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		fmt.Printf("%s\n", err)
		os.Exit(1)
	}

	for {
		file := fmt.Sprintf(".\\%s\\app测试_%s.txt", dir, time.Now().Format("2006.01.02_15_04_05"))
		_, err := os.Create(file)
		if err != nil {
			os.Exit(1)
		}
		time.Sleep(3 * time.Second)
	}
}
```

# 4 解决方案
## 4.1 方案一 使用sc命令注册服务（不推荐）

- <font size=4 color=Gold>**相关命令**</font>
    - <font size=4 color=Aqua>**注意：这些命令的执行需要以管理员的方式打开CMD**</font>
    - <font size=4 color=Aqua>**注册服务**</font>：`sc create ceshi binpath= D:\Project\ceshi\app.exe type= own start= auto displayname= ceshi`，<font size=4 color=Fuchsia>**注意等号后面的空格**</font>
    - <font size=4 color=Aqua>**删除服务**</font>：`sc delete ceshi`
    - <font size=4 color=Aqua>**启动服务**</font>：`sc start ceshi`
    - <font size=4 color=Aqua>**查看服务**</font>：`sc query ceshi`
- <font size=4 color=Gold>**测试过程**</font>
    - <font size=4 color=Aqua>**测试日志**</font>
        ```sh
        C:\Windows\system32>
        C:\Windows\system32>
        C:\Windows\system32>sc create ceshi binpath= D:\Project\ceshi\app.exe type= own start= auto displayname= ceshi
        [SC] CreateService 成功

        C:\Windows\system32>sc query ceshi

        SERVICE_NAME: ceshi
                TYPE               : 10  WIN32_OWN_PROCESS
                STATE              : 1  STOPPED
                WIN32_EXIT_CODE    : 1077  (0x435)
                SERVICE_EXIT_CODE  : 0  (0x0)
                CHECKPOINT         : 0x0
                WAIT_HINT          : 0x0

        C:\Windows\system32>sc start ceshi
        [SC] StartService 失败 1053:

        服务没有及时响应启动或控制请求。


        C:\Windows\system32>
        C:\Windows\system32>sc query ceshi

        SERVICE_NAME: ceshi
                TYPE               : 10  WIN32_OWN_PROCESS
                STATE              : 1  STOPPED
                WIN32_EXIT_CODE    : 0  (0x0)
                SERVICE_EXIT_CODE  : 0  (0x0)
                CHECKPOINT         : 0x0
                WAIT_HINT          : 0x7d0

        C:\Windows\system32>
        ```
- <font size=4 color=Gold>**弊端**</font>
    - <font size=4 color=Aqua>**这种方式注册的服务需要符合一定的规范，否则服务即使能够成功，启动服务时也极有可能报错**</font>
    

## 4.2 方案二  使用WinSW包装应用程序为服务

- <font size=4 color=Gold>**步骤**</font>
    - <font size=4 color=Aqua>**第一步：下载`WinSW`软件**</font>：[github下载链接](https://github.com/winsw/winsw/releases)
    - <font size=4 color=Aqua>**第二步：把`WinSW`复制到需要启动应用的目录位置，然后重命名，譬如我这里重命名为`app-server.exe`**</font>
        - 重命名的目的时为了后面书写命令更加方便
    - <font size=4 color=Aqua>**第三步：在目录中新建和第二步同名的`yml`配置文件，譬如我这里就是`app-server.xml`**</font>
        - 当然也可以使用`XML`语法配置文件，`WinSW`都支持。[YAML配置参考链接](https://github.com/winsw/winsw/blob/v2.12.0/doc/yamlConfigFile.md)，[XML配置参考链接](https://github.com/winsw/winsw/blob/v2.12.0/doc/xmlConfigFile.md)

        ```yaml
        # 服务ID名称（唯一）
        id: App-Server
        # 服务显示名称
        name: App-Server
        # 服务的描述信息
        description: 测试APP应用程序
        # 环境变量设置
        env:
        - name: "HOME"
          value: "%BASE%"
        # 要执行的可执行文件
        executable: "%BASE%/app.exe"
        # 可执行文件传递的参数
        # server: '%BASE%\data'
        log:
        mode: roll-by-size
        logpath: "%BASE/log%"
        sizeThreshold: 10240
        keepFiles: 8
        ```
        - <font size=4 color=Fuchsia>**注意：这里`XML`配置文件的名字必须和`WinSW`文件名相同，因为`WinSW`启动之后会去寻找同名的配置文件**</font>
    -  <font size=4 color=Aqua>**当前环境如下**</font>
        ```sh
        PS D:\Project\ceshi> ls

            目录: D:\Project\ceshi

        Mode                 LastWriteTime         Length Name
        ----                 -------------         ------ ----
        -a----        2024-03-19     21:59       18243033 app-server.exe
        -a----        2024-03-19     22:42            859 app-server.xml
        -a----        2024-03-19     21:13        4233216 app.exe
        ```
    - <font size=4 color=Aqua>**第四步：注册服务。以管理员身份启动CMD，进入到上述目录，执行`install`命令，譬如我这里为：`app-server install`**</font>
        ```sh
        D:\Project\ceshi>app-server install
        2024-03-19 22:58:45,521 INFO  - Installing service 'App-Server (App-Server)'...
        2024-03-19 22:58:45,547 INFO  - Service 'App-Server (App-Server)' was installed successfully.

        D:\Project\ceshi>
        ```
    - <font size=4 color=Aqua>**第五步：按下`win + r`，然后输入`services.msc`，打开服务页面，查看列表中是否存在前面注册的服务**</font>
        - ![](https://gouster-cloud-note.oss-cn-chengdu.aliyuncs.com/Blog/工具/windows/vx_images/372850323240360.png)
    - <font size=4 color=Aqua>**第六步：启动服务。以管理员身份启动CMD，进入到上述目录，执行`start`命令，譬如我这里为：`app-server start`**</font>
        ```sh
        D:\Project\ceshi>app-server start
        2024-03-19 23:05:24,026 INFO  - Starting service 'App-Server (App-Server)'...
        2024-03-19 23:05:24,341 INFO  - Service 'App-Server (App-Server)' started successfully.

        D:\Project\ceshi>
        ```
        - ![](https://gouster-cloud-note.oss-cn-chengdu.aliyuncs.com/Blog/工具/windows/vx_images/361423907240361.png)
    - <font size=4 color=Aqua>**第七步：校验服务是否真正启动，是否异常退出。只要发现文件在不同的创建，就说明当前配置没有问题。**</font>
        ```sh
        PS D:\Project\ceshi> tree /f
        卷 软件 的文件夹 PATH 列表
        卷序列号为 D32A-2561
        D:.
        │  app-server.exe
        │  app-server.xml
        │  app.exe
        │
        ├─app-logs
        │      app测试_2024.03.20_07_31_30.txt
        │      app测试_2024.03.20_07_31_33.txt
        │      app测试_2024.03.20_07_31_36.txt
        │      app测试_2024.03.20_07_31_39.txt
        │      app测试_2024.03.20_07_31_42.txt
        │      app测试_2024.03.20_07_31_45.txt
        │
        └─logs
                app-server.err.log
                app-server.out.log
                app-server.wrapper.log
        ```
    - <font size=4 color=Aqua>**第八步：查看服务状态。以管理员身份启动CMD，进入到上述目录，执行`status`命令，譬如我这里为：`app-server status`**</font>
        ```sh
        D:\Project\ceshi>app-server status
        Started

        D:\Project\ceshi>
        ```
- <font size=4 color=Gold>**WinSW命令说明**</font>
    - <font size=4 color=Aqua>**install**</font>：注册服务
    - <font size=4 color=Aqua>**uninstall**</font>：卸载服务
    - <font size=4 color=Aqua>**start**</font>：启动服务，启动服务之前，该服务必须已经安装
    - <font size=4 color=Aqua>**stop**</font>：停止服务
    - <font size=4 color=Aqua>**stopwait**</font>：停止服务，直到服务退出，此命令才返回
    - <font size=4 color=Aqua>**restart**</font>：重启服务
    - <font size=4 color=Aqua>**status**</font>：查看服务状态

