普通推送
git add .                           # 1. 收集：将所有修改添加到暂存区
git commit -m "初次上传"        # 2. 存档：生成新版本
git push                            # 3. 上传：推送到 GitHub

版本管理
git add .                           # 1. 收集：将所有修改添加到暂存区
git commit -m "增加DL-IAPS路径平滑"        # 2. 存档：生成新版本
git log                             # 3.查看历史，确认最新 commit 没问题
git tag -a v2.0 -m "增加DL-IAPS路径平滑"       # 4.给“当前的最新提交”贴上标签
git push origin v2.0                # 5.单独推送这个标签到远程

版本回退
git log --oneline
git clone -b 8c1963b git@github.com:kimkiihoo/vehicle_planning.git  #-b 后面可以是分支名，也可以是 Tag（标签）名。

当场回退
git log --oneline                   # 查看简略历史，找到版本号 (如 a1b2c3)
git reset --hard 8c1963b             # 彻底删除 a1b2c3 之后的所有修改！慎用！
git push -f   



首次提交
git init
git branch -M main
git add .
git commit -m "首次提交"
git remote add origin git@github.com:kimkiihoo/vehicle_planning.git
git push -u origin main   #-u或-f