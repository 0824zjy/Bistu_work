import os
import argparse
import random

def read_lines(txt_path):
    with open(txt_path, "r") as f:
        lines = [x.strip() for x in f.readlines()]
    return [x for x in lines if x]

def write_lines(lines, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for x in lines:
            f.write(x + "\n")

def parse_ratio(ratio_str):
    # ratio_str like "1:1" meaning real:gen
    a, b = ratio_str.split(":")
    return int(a), int(b)

def sample_list(pool, n, rng, replace=False):
    if n <= 0:
        return []
    if replace:
        return [pool[rng.randrange(len(pool))] for _ in range(n)]
    if n >= len(pool):
        return pool[:]  # 不够就全取（也可以改成自动replace=True）
    return rng.sample(pool, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_txt", type=str, required=True,
                    help="真实集txt，如 train_5p.txt / train_all.txt（行=ISIC_000xxxx stem）")
    ap.add_argument("--gen_txt", type=str, required=True,
                    help="某一种生成方法的全量txt，如 train_exp_cn_only.txt 或 train_exp_stage1_2.txt（行=stem 如 b-000...）")
    ap.add_argument("--tag", type=str, required=True,
                    help="生成方法标签，用于输出文件命名，如 cn_only 或 stage1_2")
    ap.add_argument("--out_dir", type=str, default="/data/zjy_work/data_txt/mix_ratio",
                    help="输出目录")
    ap.add_argument("--ratios", type=str, default="1:1,1:2,2:1,3:1",
                    help="要生成的 real:gen 比例列表，逗号分隔")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--replace", action="store_true",
                    help="若生成样本不足，是否允许有放回抽样")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    real_list = read_lines(args.real_txt)
    n_real = len(real_list)
    if n_real == 0:
        raise ValueError(f"real_txt 为空: {args.real_txt}")

    gen_pool = read_lines(args.gen_txt)
    # 去重（保持可复现：排序）
    gen_pool = sorted(set(gen_pool))
    n_gen_total = len(gen_pool)
    if n_gen_total == 0:
        raise ValueError(f"gen_txt 为空: {args.gen_txt}")

    base_real = os.path.splitext(os.path.basename(args.real_txt))[0]
    base_gen  = os.path.splitext(os.path.basename(args.gen_txt))[0]

    print(f"[INFO] real: {args.real_txt} -> {n_real}")
    print(f"[INFO] gen : {args.gen_txt} -> {n_gen_total} unique")
    print(f"[INFO] tag : {args.tag}")
    print(f"[INFO] out : {args.out_dir}")

    for r in [x.strip() for x in args.ratios.split(",") if x.strip()]:
        rk, gk = parse_ratio(r)
        n_gen_need = int(round(n_real * (gk / rk)))

        selected = sample_list(gen_pool, n_gen_need, rng, replace=args.replace)
        if args.shuffle:
            rng.shuffle(selected)

        out_txt = os.path.join(
            args.out_dir,
            f"{base_real}__{args.tag}__real{rk}_gen{gk}__nreal{n_real}_ngen{len(selected)}.txt"
        )
        write_lines(selected, out_txt)

        print(f"[OK] {r} -> need_gen={n_gen_need}, got_gen={len(selected)}")
        print(f"     {out_txt}")

if __name__ == "__main__":
    main()



"""
python /data/zjy_work/data_txt/generate_mix_ratio_txts.py \
  --real_txt /data/zjy_work/data_txt/train_20p.txt \
  --gen_txt  /data/zjy_work/data_txt/train_exp_cn_only.txt \
  --tag cn_only \
  --out_dir /data/zjy_work/data_txt/mix_ratio \
  --ratios 1:1,1:2,2:1,3:1 \
  --seed 2026 \
  --shuffle

"""
"""
python /data/zjy_work/data_txt/generate_mix_ratio_txts.py \
  --real_txt /data/zjy_work/data_txt/train_20p.txt \
  --gen_txt  /data/zjy_work/data_txt/train_exp_stage1_2.txt \
  --tag stage1_2 \
  --out_dir /data/zjy_work/data_txt/mix_ratio \
  --ratios 1:1,1:2,2:1,3:1 \
  --seed 2026 \
  --shuffle
"""
