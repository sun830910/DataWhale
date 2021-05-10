# 坑

1. DBeaver，数据库连接后执行sql脚本出错：

   - 打开mysql命令行，一行一行执行，定位问题
   - 报错：[MySQL创建用户时提示“Operation CREATE USER failed for   XXX”](https://blog.csdn.net/u011870022/article/details/111411031)
     - drop user 'bluewhale';
     - create user 'bluewhale'@'%' identified by 'bluewhale';
   - [mysql添加用户及授权](https://blog.csdn.net/qq_25109517/article/details/108708097)

2. 直接windows安装python3.8，**加入Path！！！**

   - 记得在项目下启动环境：**pipenv shell！！！**

3. 本机出现中文，django编译出错！

   - 改本机名，全为英文！！！（所有文件都是，以后命名都用英文+数字）
   - ![1620628745079](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1620628745079.png)

4. 更改本机名后，mysql连不上

   - **管理员身份**（右键选项）启动cmd，在mysql安装bin目录下， 运行"mysqld --install" 
   - net start mysql ，启动mysql
   - mysql -u root -p 输入密码，成功启动；（同时检查mysql workbench）一般就没有问题了，DBeaver也可以正常连接！
   - 注，有可能用户名密码错误，也这样启动， 通过“set password=password('root')”修改密码。此处将root密码设置为root 

5. 安装nodejs

   - 一路下一步， 执行 node -v 和 npm -v 分别查看node和npm的版本号，正常即成功

   - [配置npm在安装全局模块时的路径和缓存cache的路径](https://blog.csdn.net/pyf09/article/details/109525954)

     - 在node.js安装目录下新建两个文件夹 node_global和node_cache，然后在cmd命令下执行如下两个命令：

       npm config set prefix "D:\Program Files\nodejs\node_global"

       npm config set cache "D:\Program Files\nodejs\node_cache"

     - 系统变量中新建一个变量名为 “NODE_PATH”， 值为“D:\Program Files\nodejs\node_modules” 

     - 用户变量中，将相应npm的路径改为：D:\Program Files\nodejs\node_global 

     - 在cmd命令下执行 npm install webpack -g 然后安装成功后可以看到自定义的两个文件夹已生效 ， 执行 npm webpack -v 可以看到所安装webpack的版本号 

6. 前后端，在各自的子目录同步依赖包并启动

   - ![1620629676740](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1620629676740.png)
   - ![1620629686642](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1620629686642.png)

7. git-windows

   - [git bash](https://blog.csdn.net/qq_37838568/article/details/81052370),[下载](https://git-scm.com/downloads)，一路下一步安装
   - [key](https://blog.csdn.net/qq_37838568/article/details/81053055),
   - 本地找好位置，右键git bash，克隆项目：git clone git@github.com:YetAnotherWebTeam/whale-web.git
   - 建立分支，提交
     - git branch
     - git checkout -b web_yx
     - git status
     - git add .
     - git commit'xxx'
     - git push origin web_yx




