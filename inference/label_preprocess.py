# label_preprocess.py
# Annotation 파일에서 선박 단일 클래스로 매핑 후 YOLO label 형식에 맞게 변환하는 코드

import os 
import glob
import xml.etree.ElementTree as ET

ann_dir = "./test/Annotations"
img_dir = "./test/images"
labels_out = "./test/labels"

os.makedirs(labels_out, exist_ok=True)

SHIP_IDX = 7

# HRSC 데이터에 등장한 ship 관련 Class_ID 전부 하나로 매핑
class_ids = [
    '000001', '100000001', '100000002', '100000003', '100000004',
    '100000005', '100000006', '100000007', '100000008', '100000009',
    '100000010', '100000011', '100000012', '100000013', '100000015',
    '100000016', '100000017', '100000018', '100000019', '100000020',
    '100000022', '100000024', '100000025', '100000026', '100000027',
    '100000028', '100000029', '100000030', '100000032',
]
CLASS_MAP = {cid: SHIP_IDX for cid in class_ids}

xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
print(f"XML 파일 개수: {len(xml_files)}")

for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_name = root.findtext("Img_FileName").strip()
    img_w = int(root.findtext("Img_SizeWidth"))
    img_h = int(root.findtext("Img_SizeHeight"))

    img_path = os.path.join(img_dir, img_name + ".bmp")
    if not os.path.exists(img_path):
        print(f"[스킵] 이미지 없음: {img_path}")
        continue

    label_path = os.path.join(labels_out, img_name + ".txt")
    lines = []
    objects = root.find("HRSC_Objects")

    if objects is not None:
        for obj in objects.findall("HRSC_Object"):
            class_id = obj.findtext("Class_ID").strip()
            if class_id not in CLASS_MAP:
                continue
            cls = CLASS_MAP[class_id]
            xmin = float(obj.findtext("box_xmin"))
            ymin = float(obj.findtext("box_ymin"))
            xmax = float(obj.findtext("box_xmax"))
            ymax = float(obj.findtext("box_ymax"))
            # YOLO format (normalized)
            cx = (xmin + xmax) / 2.0 / img_w
            cy = (ymin + ymax) / 2.0 / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

print("YOLO 라벨 변환 완료")
print("생성된 라벨 수:", len(glob.glob(os.path.join(labels_out, "*.txt"))))