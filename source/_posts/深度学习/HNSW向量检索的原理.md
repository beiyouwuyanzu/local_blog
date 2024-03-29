---
title: HNSW向量检索的原理
date: 2023-09-05 23:26:22
---


HNSW（Hierarchical Navigable Small World）是一种高效的向量检索算法，它可以在大规模向量集上实现快速近似最近邻搜索。这个算法最初由Yu. A. Malkov和D. A. Yashunin在2016年的论文《Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs》中提出。

HNSW算法的核心原理是使用一个层次化的图结构，通过在不同层级上组织向量数据来加速最近邻搜索。算法主要分为两个阶段：构建阶段和搜索阶段。

1. 构建阶段：
在构建阶段，HNSW算法会创建一个层次化的图结构，其中包含了待检索的向量数据。算法的具体步骤如下：

a. 初始化：从数据集中随机选择一个向量作为起始点，创建一个包含该向量的图，并将其作为第0层的顶点。

b. 添加向量：对于每个待检索的向量，首先选择一个起始顶点来开始搜索。搜索的起始点在图中根据一定的策略进行选择（例如，随机选择、轮盘赌选择等）。然后，从起始点开始，通过计算欧几里得距离（或其他相似度度量）来寻找该向量的近邻。一旦找到近邻，就将该向量添加到图中，并在相应的层级上进行连接。

c. 构建连接：HNSW算法中的“小世界”特性是通过添加一些额外的连接来实现的。这些额外的连接允许向量在图中跳跃到不同的层级，从而增加了搜索的效率。连接的建立需要满足一定的条件，以保持图的平衡性和紧密性。

d. 层级划分：为了实现高效的搜索，HNSW算法会在图中定义多个层级。每个层级都包含一部分数据，并且连接只能在同一层级或相邻层级之间建立。

2. 搜索阶段：
在搜索阶段，HNSW算法利用构建阶段得到的图结构来快速近似地找到一个向量的最近邻。搜索的过程如下：

a. 初始化：从图的顶层（最高层级）开始搜索，将待查询的向量作为搜索起始点。

b. 向下导航：根据某种启发式方法，在当前层级中向下导航，寻找距离查询向量较近的顶点。

c. 向上导航：一旦在较低层级找到潜在的近邻，就会在图的更高层级上进一步验证和改善近邻的选择。

d. 近似结果返回：在搜索过程中，可以维护一个有限的候选集合，记录搜索过程中找到的近邻。最终，从候选集合中选择距离查询向量最近的向量作为近似的最近邻结果。

通过这种层次化的结构和小世界的连接策略，HNSW能够在大规模向量数据集上快速、高效地进行近似最近邻搜索，尤其适用于高维向量数据的检索任务。