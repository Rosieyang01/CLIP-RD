import os, json, torch

def karpathy_split_flickr(karpathy_json, out_dir):
    data = json.load(open(karpathy_json, "r", encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)

    splits = {"train": {}, "val": {}, "test": {}}

    for it in data["images"]:
        sp = it["split"]
        if sp not in splits:
            continue

        fname = it["filename"]
        caps = [s["raw"].strip() for s in it["sentences"]]
        splits[sp][fname] = caps

    for sp, caps_dict in splits.items():
        keys = sorted(caps_dict.keys())

        with open(os.path.join(out_dir, f"{sp}_img_keys.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(keys) + "\n")

        torch.save(caps_dict, os.path.join(out_dir, f"{sp}_captions.pt"))
        print(sp, "num_images =", len(keys))

if __name__ == "__main__":
    karpathy_split_flickr("path/to/dataset_flickr32k.json", "./flickr_karpathy_eval")