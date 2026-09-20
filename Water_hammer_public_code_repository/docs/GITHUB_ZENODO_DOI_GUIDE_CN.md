# 在 GitHub 上发布代码并通过 Zenodo 获得 DOI

## 先说明一个关键点

GitHub 本身不签发 DOI。通常做法是：**GitHub 保存持续更新的代码，Zenodo 归档某一个 GitHub Release，并为该不可变版本签发 DOI。**

Zenodo 通常会给出两个标识：

- **版本 DOI**：只对应某一次 release，例如 `v1.0.0`；
- **概念 DOI（Concept DOI）**：汇总该软件的所有版本，适合在需要指向“持续更新的软件项目”时使用。

论文最好引用实际支撑稿件结果的**版本 DOI**，因为它对应固定代码快照。

## 一、投稿双盲审稿阶段

如果稿件采用双盲审稿，不建议直接在匿名稿中填写个人 GitHub 仓库地址，因为账号、提交历史和 `CITATION.cff` 都可能暴露作者身份。

建议流程：

1. 先在本地保留本代码包，检查其中没有姓名、单位、个人邮箱和本机/服务器绝对路径。
2. 若投稿系统强制要求 URL，可将审稿快照上传至支持私密记录和匿名访问链接的数据仓库，向审稿人提供匿名链接。
3. 等稿件录用或期刊允许解除匿名后，再公开 GitHub 仓库、连接 Zenodo 并创建正式 DOI。
4. 匿名稿中的数据可用性声明不要写尚未生成的 DOI，也不要写会暴露身份的个人 GitHub URL。

## 二、建立 GitHub 仓库

1. 登录 GitHub，右上角选择 **New repository**。
2. 仓库名建议使用简洁英文，例如：

   `characteristic-pinn-water-hammer-fssi`

3. 双盲阶段先设为 **Private**；解除匿名后再设为 **Public**。
4. 不要让 GitHub 再自动生成 README、`.gitignore` 或 LICENSE，以免与本文件夹中的版本冲突。
5. 在本文件夹中执行：

   ```bash
   git init
   git add .
   git commit -m "Prepare reproducible code release"
   git branch -M main
   git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
   git push -u origin main
   ```

6. 检查网页端没有上传 `outputs/`、权重、日志、PDF、第三方数据或作者不希望公开的文件。

## 三、补全引用元数据

1. 将 `CITATION.cff.template` 复制为 `CITATION.cff`。
2. 填写最终作者姓名、GitHub 地址和发布日期。
3. 第一次发布前 DOI 仍未知，可以暂时删除模板中的 `doi:` 行；Zenodo 归档后再补回 DOI，并在后续 release 中更新。
4. 确认 LICENSE 符合作者的共享意愿。本包暂按 MIT License 准备。

## 四、连接 Zenodo

1. 登录 <https://zenodo.org/>，建议使用 GitHub 账号授权登录。
2. 打开 Zenodo 的 GitHub 集成页面并授权 Zenodo 访问 GitHub。
3. 在仓库列表中找到目标仓库，打开归档开关。
4. 回到 GitHub，创建正式版本：

   - 点击 **Releases** → **Draft a new release**；
   - 新建 tag，例如 `v1.0.0`；
   - Release title 可写 `v1.0.0 – manuscript submission code`；
   - 简要说明该版本对应哪一版稿件；
   - 点击 **Publish release**。

5. Zenodo 接收到 GitHub release 后会自动建立归档记录并分配 DOI。
6. 在 Zenodo 页面核对标题、作者顺序、单位、关键词、关联论文题目、许可证和版本号；缺失信息应及时补全。
7. 复制 Zenodo 页面显示的版本 DOI，例如：

   `https://doi.org/10.5281/zenodo.xxxxxxx`

8. 把真实 DOI 写回 `CITATION.cff` 和 README，再创建一个小版本 release（如 `v1.0.1`）用于保存更新后的引用信息。论文中仍可引用实际支撑结果的 `v1.0.0` 版本 DOI。

## 五、投稿系统中填写什么

当正式 DOI 已公开且不会破坏匿名要求时，在“DOI or other location of your data”一栏填写：

`https://doi.org/10.5281/zenodo.xxxxxxx`

不要只填写 GitHub 首页地址。DOI 指向固定归档，更适合长期引用。

## 六、后续更新

代码改动后不要覆盖已归档的 `v1.0.0`。应提交新 commit，并创建 `v1.1.0` 或 `v1.0.1` release。Zenodo 会为新版本生成新的版本 DOI，同时把各版本归入同一个概念 DOI。
