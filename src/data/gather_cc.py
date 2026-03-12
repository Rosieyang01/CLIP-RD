import requests
import os
import multiprocessing as mp
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
import pickle
import sys


def grab(args):
    """
    Download a single image from the TSV.
    """
    uid, split, line, root = args   # ★ ROOT 대신 root 인자로 받음
    try:
        caption, url = line.split("\t")[:2]
    except:
        print("Parse error")
        return

    dst = f"{root}/{split}/{uid % 1000}/{uid}.jpg"

    if os.path.exists(dst):
        print("Finished", uid)
        return uid, caption, url

    try:
        dat = requests.get(url, timeout=20)
        if dat.status_code != 200:
            print("404 file", url)
            return

        im = Image.open(BytesIO(dat.content))
        im.thumbnail((512, 512), PIL.Image.BICUBIC)
        if min(*im.size) < max(*im.size) / 3:
            print("Too small", url)
            return

        # ★ 디렉터리 자동 생성 추가
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        im.save(dst)

        try:
            o = Image.open(dst)
            o = np.array(o)
            print("Success", o.shape, uid, url)
            return uid, caption, url
        except:
            print("Failed", uid, url)

    except Exception as e:
        print("Unknown error", e)
        pass


if __name__ == "__main__":
    root = sys.argv[1]  # ★ ROOT를 지역변수로 받음

    if not os.path.exists(root):
        os.mkdir(root)
        os.mkdir(os.path.join(root, "train"))
        os.mkdir(os.path.join(root, "val"))
        for i in range(1000):
            os.mkdir(os.path.join(root, "train", str(i)))
            os.mkdir(os.path.join(root, "val", str(i)))

    # ★ 멀티프로세스 너무 많으면 서버 멈춤 → 16~24로 제한
    num_workers = min(mp.cpu_count(), 16)
    p = mp.Pool(num_workers)

    for tsv in sys.argv[2:]:
        print("Processing file", tsv)
        assert 'val' in tsv.lower() or 'train' in tsv.lower()
        split = 'val' if 'val' in tsv.lower() else 'train'

        with open(tsv, "r") as f:
            lines = f.read().split("\n")

        # ★ ROOT를 grab에 직접 전달
        results = p.map(grab, [(i, split, x, root) for i, x in enumerate(lines)])

        out = open(tsv.replace(".tsv", ".csv"), "w")
        out.write("title\tfilepath\n")

        for row in results:
            if row is None:
                continue
            id, caption, url = row
            fp = os.path.join(root, split, str(id % 1000), str(id) + ".jpg")
            image_local_path = os.path.join(split, str(id % 1000), str(id) + ".jpg")
            if os.path.exists(fp):
                out.write("%s\t%s\n" % (caption, image_local_path))
            else:
                print("Drop", id)
        out.close()

    p.close()