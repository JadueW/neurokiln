---
layout: home

hero:
  name: NeuroKiln
  text: 脑炉
  tagline: 可复现、防泄漏的 BCI 算法评测与调参平台
  image:
    src: /banner.png
    alt: NeuroKiln
  actions:
    - theme: brand
      text: 阅读用户手册
      link: /user-manual
    - theme: alt
      text: 五分钟上手
      link: /user-manual

features:
  - title: 协议优先
    details: 被试内、跨 session、LOSO 三种协议；预处理、特征选择与调参全部嵌套在训练折内。
  - title: 丹方 DAG
    details: 30+ 节点自由组合，支持命名端口、滤波器组并行与多解码器融合。
  - title: 可审计调参
    details: 灰狼优化 + Pareto 存档，代理折筛选后完整复评，失败原因逐条记录。
  - title: 一键复现
    details: 每次实验记录数据签名、划分、参数与环境；导出复现包并按容差重跑比对。
  - title: 模型透视
    details: CSP / xDAWN / 卷积核内在参数与 topomap、遮挡与频带归因、Captum 梯度、忠实性检验、知识闭包审计，以及带支持计数的跨折聚合。
  - title: 可恢复队列
    details: 后台任务持久化，重启自动重排，实验与解释副作用幂等；失败重试上限可配置，错误码可直接映射中英文提示。
  - title: 论文级导出
    details: HTML、Excel、Word、LaTeX 与 GB/T 7714 参考文献，直接进入论文写作流程。
  - title: 本地优先
    details: 单机 SQLite、本地账号、Docker 一条命令启动；数据不离开你的机器。
---

## 这是什么

NeuroKiln 把「数据版本 → 协议划分 → 丹方 → 指标与统计 → 报告/复现包」收敛成一条固定流水线。

> 同一数据签名 + 同一协议 + 同一种子集合，才允许同表排名。

如果你正在做 MI、P300 或 SSVEP 的离线算法比较，NeuroKiln 可以帮你避免数据泄漏、结果不可比和复现失败。

## 快速开始

```bash
docker compose up -d --build
```

打开 <http://127.0.0.1:8100>，创建本机管理员账号，然后点击「五分钟上手」。

完整安装、CLI、插件与安全说明见[用户手册](/user-manual)。
