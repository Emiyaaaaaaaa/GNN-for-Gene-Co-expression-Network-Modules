import numpy as np
import requests
import pandas as pd
from typing import List, Dict, Tuple, Optional


protein_relations = []
header = ['protein1', 'protein2', 'combined_score']
file_path = r'F:\项目模型\基因转录\基因转录\datee\9031.protein.links.v12.0.txt'
i_num = 0
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:  # 逐行迭代，内存友好
        if not line:  # 跳过空行
            continue
        if line.startswith('#'):  # 跳过注释行（STRING文件表头通常以#开头）
            continue
        # 跳过表头行（与目标表头完全匹配时）
        parts = line.split()
        if parts == header:
            continue

        if int(parts[2]) < 500:
            continue
        protein_relations.append(parts)
        i_num += 1
        print(i_num, parts)

        if len(protein_relations) == 1000:
            break

def extract_protein_ids(relations: List[str]) -> List[str]:
    """从蛋白质关系数据中提取所有唯一的蛋白质ID"""
    protein_ids = set()
    for line in relations:
        parts = line
        if len(parts) >= 2:
            protein_ids.add(parts[0])
            protein_ids.add(parts[1])
    return list(protein_ids)


def map_protein_to_gene(protein_list: List[str], species: int = 9031) -> pd.DataFrame:
    """
    通过STRING API将蛋白质ID映射为基因ID和基因符号
    返回：DataFrame(protein_id, gene_id, gene_symbol)
    """
    string_api_url = "https://string-db.org/api"
    output_format = "tsv"
    method = "get_string_ids"

    # 分批处理蛋白质ID，避免请求过长
    batch_size = 200
    all_results = []

    for i in range(0, len(protein_list), batch_size):
        batch = protein_list[i:i + batch_size]

        # API参数
        params = {
            "identifiers": "\r".join(batch),  # 蛋白质ID列表，用回车分隔
            "species": species,  # 物种的Taxonomy ID
            "limit": 1,  # 仅返回最佳匹配
            "echo_query": 1,  # 保留原始查询ID
            "caller_identity": "protein_to_gene_mapping"  # 身份标识
        }

        # 发送请求
        request_url = "/".join([string_api_url, output_format, method])
        try:
            response = requests.post(request_url, data=params, timeout=30)
            response.raise_for_status()  # 检查请求是否成功

            # 解析TSV格式结果
            lines = response.text.strip().split("\n")
            if len(lines) > 1:  # 确保有数据返回
                headers = lines[0].split("\t")
                data = [line.split("\t") for line in lines[1:]]
                batch_df = pd.DataFrame(data, columns=headers)
                all_results.append(batch_df)

        except requests.exceptions.RequestException as e:
            print(f"批次 {i // batch_size + 1} 请求失败: {e}")
            # 可以考虑添加重试逻辑

    if not all_results:
        raise Exception("所有批次请求均失败，无法获取映射数据")

    # 合并所有批次结果
    df = pd.concat(all_results, ignore_index=True)

    # 提取关键列
    df["protein_id"] = df["queryItem"]
    df["string_id"] = df["stringId"]
    df["gene_symbol"] = df["preferredName"]

    # 转换为Ensembl基因ID
    def extract_gene_id(string_id: str) -> Optional[str]:
        if isinstance(string_id, str) and "ENSGALP" in string_id:
            return string_id.replace("ENSGALP", "ENSGALG")
        return None

    df["gene_id"] = df["string_id"].apply(extract_gene_id)

    return df[["protein_id", "gene_id", "gene_symbol"]]


def convert_to_gene_relations(protein_relations: List[str],
                              protein_to_gene: Dict[str, str]) -> List[Tuple[str, str, int]]:
    """
    将蛋白质-蛋白质关系转换为基因-基因关系
    返回：[(gene_id1, gene_id2, score), ...]
    """
    gene_relations = []
    for line in protein_relations:
        parts = line
        if len(parts) != 3:
            print(f"跳过格式不正确的行: {line}")
            continue

        p1, p2, score = parts
        g1 = protein_to_gene.get(p1)
        g2 = protein_to_gene.get(p2)

        # 确保两个蛋白质都成功映射到基因，并且基因ID不同
        if g1 and g2 and g1 != g2:
            try:
                score_value = int(score)
                gene_relations.append((g1, g2, score_value))
            except ValueError:
                print(f"分数转换失败: {score}")

    return gene_relations


def gens_map():
    # 1. 提取蛋白质ID
    protein_list = extract_protein_ids(protein_relations)
    print(f"提取了 {len(protein_list)} 个唯一蛋白质ID")

    # 2. 执行映射
    print("正在通过STRING API进行蛋白质到基因的映射...")
    mapping_df = map_protein_to_gene(protein_list)

    # 创建蛋白质到基因ID的映射字典
    protein_to_gene = dict(zip(mapping_df["protein_id"], mapping_df["gene_symbol"]))

    # 统计映射成功的比例
    mapped_count = sum(1 for p in protein_list if protein_to_gene.get(p))
    print(f"成功映射 {mapped_count}/{len(protein_list)} 个蛋白质ID ({mapped_count / len(protein_list):.2%})")

    # 3. 转换蛋白质-蛋白质关系为基因-基因关系
    gene_relations = convert_to_gene_relations(protein_relations, protein_to_gene)
    print(f"转换得到 {len(gene_relations)} 条基因-基因关系")

    # 4. 输出结果
    print("\n基因-基因关系（基因ID）：")
    gen_gen_edges = []
    for g1, g2, score in gene_relations:
        print(f"{g1} {g2} {score}")
        gen_gen_edges.append([g1, g2, score])

    return gen_gen_edges


# if __name__ == "__main__":
#     gens_map()