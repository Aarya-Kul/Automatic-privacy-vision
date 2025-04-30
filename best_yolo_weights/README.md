# Best weights

Currently, these are weights from finetuning
the pre-trained `yolo11s-seg.pt` model.

# Hyperparameters

```json
{
  "epochs": 50,
  "imgsz": 640, # image size
  "batch": 16, # batch size
  "lr0": 0.01, # learning_rate
  "momentum": 0.937,
  "weight_decay": 0.0005,
  "degrees": 0.25,
  "scale": 0.5,
  "perspective": 0.0001,
  "fliplr": 0, # better performance without
  "erasing": 0, # better performance without
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "workers": 4,
  "project": "./runs/",
  "name": args.dir_name,
}
```
