# 如何将本地文件提交到GitHub

下面是将本地文件提交到GitHub的完整步骤指南：

## 前提条件
- 已安装Git
- 已在GitHub上创建账号
- 已在GitHub上创建空仓库

## 步骤1：初始化本地Git仓库（如果尚未初始化）
```bash
# 在项目根目录下执行
cd D:\graduate
git init
```

## 步骤2：配置Git用户名和邮箱
```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

## 步骤3：将文件添加到暂存区
```bash
# 添加单个文件
git add 文件名

# 添加所有文件（推荐）
git add .
```

## 步骤4：提交更改到本地仓库
```bash
git commit -m "提交说明，描述你做了什么更改"
```

## 步骤5：连接到GitHub远程仓库
```bash
# 替换为你的GitHub仓库URL
git remote add origin https://github.com/你的用户名/仓库名.git
```

## 步骤6：将本地提交推送到GitHub
```bash
git push -u origin main
# 或使用master分支（老版本GitHub默认分支名）
git push -u origin master
```

## 常见问题解决

### 如果远程仓库已有内容
如果GitHub上的仓库不是空的，可能需要先拉取（pull）远程内容：
```bash
git pull --rebase origin main
git push -u origin main
```

### 查看当前状态
```bash
git status
```

### 查看提交历史
```bash
git log
```

## 注意事项
1. 不要将敏感信息（如API密钥、密码等）提交到GitHub
2. 考虑创建.gitignore文件来忽略不需要提交的文件
3. 定期推送你的更改，确保备份

按照以上步骤操作，你应该能够成功将本地文件提交到GitHub仓库。