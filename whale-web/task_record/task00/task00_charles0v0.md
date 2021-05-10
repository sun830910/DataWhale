按照教程来基本可以完成。

## 遇到的问题
1. 安装 Dbeaver的时候记得勾选include java， 否则会不安装java runtime, 在没有合适的jre的机器上是无法运行的
2. Dbeaver 连接Mysql可能会报没有驱动的问题，安装即可。因为外网下载速度较慢，可以设置下载代理。
3. 在pipenv 安装后端环境的时候，提示缺少cryptography，使用pipenv install cryptography 即可。此项操作会之后PipFile和PipFile.lock会被修改

