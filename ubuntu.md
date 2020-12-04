# ubuntu 安装操作手册

## 允许远程登录
https://www.cnblogs.com/alley-wmq/p/12872071.html

## 固定ip
https://www.cnblogs.com/blogxu/p/ubuntu.html

```
network:
  ethernets:
    eno2:
      addresses: [10.13.14.123/24]
      gateway4: 10.13.14.1
      nameservers:
              addresses: [144.144.144.144,8.8.8.8]
      dhcp4: false
  version: 2
```


`sudo netplan apply`

## SSH免密码登录(证书登录)

证书创建时都直接用默认信息不设置密码(一路回车)
```
ssh-keygen -t rsa
```
证书创建完成以后会在`/Users/{userName}/.ssh/`目录下{userName}是电脑的管理员用户名.
```
lorne@lorne .ssh % ls
id_rsa      id_rsa.pub  known_hosts
lorne@lorne .ssh % 
```

将pub证书信息添加到部署服务器下并授权.   

上传证书 
```
scp id_rsa.pub root@local.api:/root/.ssh/
```
输入密码为`codingapi.`

授权证书登录

```
ssh root@lcoal.api
cd /root/.ssh
cat id_rsa.pub >> authorized_keys
```

## 挂在硬盘
https://blog.51cto.com/12348890/2092339

## 源更新
### 更新镜像源(ubuntu):
https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/

```
sudo apt-get update
sudo apt-get upgrade
```

### 更新镜像源(anaconda):
https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
### 更新镜像源(pip):
https://mirrors.tuna.tsinghua.edu.cn/help/pypi/


## cuda 10.1 安装
https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
## cuda 卸载教程
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-tk-and-driver

## conda 安装教程
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

##  kvm cockpit  cockpit-machines
https://www.cnblogs.com/johnsonjie/p/10876417.html 
https://my.oschina.net/u/3585265/blog/3154196
 
### 安装 libvirtd 
  libvirtd 是虚拟化的平台，类似KVM
```
sudo apt update
sudo apt install qemu qemu-kvm libvirt-bin bridge-utils virt-manager
sudo systemctl start libvirtd.service
sudo systemctl enable libvirtd.service
```
配置网桥(yml 空格必须缩紧两位)
```
# This is the network config written by 'subiquity'
network:
  ethernets:
    eno2:
      dhcp4: false
  bridges:
    kvmbr0:
      interfaces: [eno2]
      dhcp4: no
      addresses: [10.13.14.123/24]
      gateway4: 10.13.14.1
      nameservers:
        addresses: [144.144.144.144,8.8.8.8]
  version: 2
```
### 安装cockpit
  cockpit 是电脑的资源管理器
```
sudo apt install cockpit cockpit-machines
sudo systemctl enable libvirtd --now
sudo systemctl enable cockpit.socket --now
```
访问地址： http://ip:9090
### virsh 指令说明
https://ubuntu.com/server/docs/virtualization-libvirt

## pycharm 利用服务器资源运行环境
https://www.bilibili.com/read/cv3097446/


## juypter notebook 
服务器开启方式：

需要提前开启gluon
```
conda activate gluon
```

--allow-root (root账户下)
```
jupyter notebook  --allow-root

```
本地电脑映射，以上myserver是远端服务器地址。然后我们可以使用 http://localhost:8888 打开运行Jupyter记事本的远端服务器myserver。我们将在下一节详细介绍如何在AWS实例上运行Jupyter记事本
```
ssh lorne.ubuntu -L 8888:localhost:8888
```