# 知识库版本管理流程

## 概述

知识库按"发布学年"标签进行版本管理。每个文档片段（chunk）在入库时会被标记 `academic_year` 字段（如 `2025-2026`），支持按学年进行增量导入、重建和回滚操作。

## 快速命令

```bash
# 查看当前所有学年
python run_import.py --list-years

# 增量导入新文件
python run_import.py --dir data/raw/新政策文件 --academic-year 2025-2026

# 统计某学年数据量
python run_rebuild.py --count 2025-2026

# 重建某学年（先删后导）
python run_rebuild.py --rebuild 2025-2026 --dir data/raw/2025-2026政策

# 回滚（删除）某学年
python run_rebuild.py --rollback 2025-2026
```

## 场景操作指南

### 场景 1：新学年政策更新（增量导入）

当新学年的政策文件发布后，需要将新文件追加到知识库，不影响旧学年数据。

```bash
# 1. 将新文件放入一个目录
mkdir data/raw/2026-2027新政策
cp 新文件1.pdf 新文件2.docx data/raw/2026-2027新政策/

# 2. 增量导入
python run_import.py --dir data/raw/2026-2027新政策 --academic-year 2026-2027

# 3. 验证
python run_import.py --list-years
# 应看到新学年及其 chunk 数量
```

**注意**：增量导入不会删除已有数据，只追加新数据。

### 场景 2：某学年政策有误（重建）

当某学年的政策文件有更新或修正时，需要先删除旧数据再重新导入。

```bash
# 1. 备份旧文件（可选）
# 2. 将修正后的文件放入目录
# 3. 重建（会先删除旧数据再导入）
python run_rebuild.py --rebuild 2025-2026 --dir data/raw/2025-2026政策修正版

# 4. 验证
python run_rebuild.py --count 2025-2026
```

**注意**：重建操作会先删除该学年的全部数据，请确认源目录包含该学年的所有文件。

### 场景 3：某学年政策已废止（回滚）

当某学年的政策完全废止不再使用时，可以删除该学年的全部数据。

```bash
# 1. 确认要删除的学年
python run_rebuild.py --count 2024-2025

# 2. 回滚（会提示确认）
python run_rebuild.py --rollback 2024-2025

# 3. 验证
python run_rebuild.py --list-years
```

## 技术细节

### 数据存储

| 存储 | 用途 | 同步方式 |
|------|------|----------|
| Milvus 向量库 | 向量检索 | 按 `academic_year` 字段过滤删除 |
| `data/processed/chunks.json` | BM25 关键词检索 | 按 `source_file` 同步增删 |

### 核心模块

- `src/versioning.py`：版本管理核心，提供 `list_academic_years()` / `delete_by_academic_year()` / `import_files()` / `rebuild_academic_year()` 等函数
- `run_import.py`：增量导入入口脚本
- `run_rebuild.py`：重建/回滚入口脚本

### 学年标签规范

- 格式：`YYYY-YYYY`，如 `2025-2026`
- 从文件名或正文中自动提取（`MetadataExtractor._extract_academic_year()`）
- 导入时通过 `--academic-year` 参数手动指定，覆盖自动提取的值

## 注意事项

1. **重建前确认**：重建操作会先删除该学年的全部数据，请确保源目录包含该学年的完整文件集
2. **回滚不可逆**：删除后无法恢复，除非有源文件可以重新导入
3. **BM25 同步**：所有写操作会同时更新 Milvus 和 `chunks.json`，确保两路检索数据一致
4. **Embedding 成本**：增量导入和重建都会调用 Embedding API，注意 Token 消耗
