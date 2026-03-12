import json, os, torch

def karpathy_split_coco(dataset_coco_json, out_dir):
    with open(dataset_coco_json, "r") as f:
        data = json.load(f)

    
    test_items = [it for it in data["images"] if it.get("split") == "test"]

    captions = {}
    img_keys = []

    for it in test_items:
        image_id = int(it["cocoid"]) 
        sents = [s["raw"] for s in it["sentences"]]
        captions[image_id] = sents
        img_keys.append(image_id)

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "test_img_keys.tsv"), "w") as f:
        for k in img_keys:
            f.write(f"{k}\n")

    
    torch.save(captions, os.path.join(out_dir, "test_captions.pt"))

if __name__ == "__main__":
    karpathy_split_coco("path/to/dataset_coco.json", "./coco_eval")