from config import cfg
from models import build_model
from torchvision.transforms import transforms
from util.TextCountingUtils import SamClipModel

import torch
from PIL import Image

cfg.merge_from_file("./config/test_bmnet+.yaml")

model = build_model(cfg)

checkpoint_path = "./checkpoints/bmnet+_pretrained/model_best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(checkpoint["model"])
model.eval()


def preprocess_image(image_path, boxes=None, text_prompt=None
                     , scale_number=20, query_transform=None, min_size=384, max_size=1584):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    r = 1.0
    if h > max_size or w > max_size:
        r = max_size / max(h, w)
    if r * h < min_size or r * w < min_size:
        r = min_size / min(h, w)
    nh, nw = int(r * h), int(r * w)
    img = img.resize((nw, nh), resample=Image.BICUBIC)

    patches = []
    scale_embedding = []
    if boxes is None and text_prompt is not None:
        patch, box = SamClipModel(SamModelPath="./checkpoints/bmnet+_pretrained/sam_vit_h_4b8939.pth"
                                  , image=img, text_prompt=text_prompt)

        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * r), int(y1 * r), int(x2 * r), int(y2 * r)

        patch = query_transform(patch) if query_transform else transforms.ToTensor()(patch)
        patches.append(patch)

        scale = (x2 - x1) / nw * 0.5 + (y2 - y1) / nh * 0.5
        scale = scale // (0.5 / scale_number)
        scale = min(scale, scale_number - 1)
        scale_embedding.append(scale)
    elif boxes is not None and text_prompt is None:
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * r), int(y1 * r), int(x2 * r), int(y2 * r)

            patch = img.crop((x1, y1, x2, y2))

            patch = query_transform(patch) if query_transform else transforms.ToTensor()(patch)
            patches.append(patch)

            scale = (x2 - x1) / nw * 0.5 + (y2 - y1) / nh * 0.5
            scale = scale // (0.5 / scale_number)
            scale = min(scale, scale_number - 1)
            scale_embedding.append(scale)
    else:
        print("Error, only one can be selected: either a box or a text prompt.")
        exit(0)
    patches = torch.stack(patches, dim=0) if patches else torch.tensor([])
    scale_embedding = torch.tensor(scale_embedding, dtype=torch.long).unsqueeze(0)

    return img, {"patches": patches, "scale_embedding": scale_embedding}


def CountByBoxPrompt():
    query_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_path = "./img.jpg"
    boxes = [[291, 234, 374, 312]]

    img, patches = preprocess_image(image_path, boxes, query_transform=query_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    patches["patches"] = patches["patches"].to(device)
    patches["scale_embedding"] = patches["scale_embedding"].to(device)

    with torch.no_grad():
        output = model(img, patches, is_train=False)

    print("Count results is : ", output.sum().item())


def CountByTextPrompt():
    query_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_path = "4882.jpg"
    text_prompt = ["strawberry"]

    img, patches = preprocess_image(image_path, text_prompt=text_prompt, query_transform=query_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    patches["patches"] = patches["patches"].to(device)
    patches["scale_embedding"] = patches["scale_embedding"].to(device)

    with torch.no_grad():
        output = model(img, patches, is_train=False)

    print("Count results is : ", output.sum().item())


if __name__ == '__main__':
    # CountByBoxPrompt()

    CountByTextPrompt()
