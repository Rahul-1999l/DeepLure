import os
import json
import csv
import cv2
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Directories
root_dir = Path(__file__).resolve().parents[1]
dataset_dir = root_dir / "sarees_dataset"
data_dir = root_dir / "data"
output_dir = root_dir / "outputs" / "dataset_inspection"

data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load all images
image_paths = sorted(list(dataset_dir.rglob("*.jpg")))
print(f"Total images found: {len(image_paths)}")

# 2. Extract SIFT features & compute initial 72 design groups (exact preservation)
sift = cv2.SIFT_create(nfeatures=600)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

descriptors = {}
image_metadata = {}

for p in image_paths:
    rel_path = str(p.relative_to(root_dir))
    with Image.open(p) as img:
        w, h = img.size
        ar = round(w / h, 2)
    
    cv_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(cv_img, None)
    descriptors[p.name] = des
    
    image_metadata[p.name] = {
        'full_path': str(p),
        'rel_path': rel_path,
        'filename': p.name,
        'width': w,
        'height': h,
        'aspect_ratio': ar
    }

# Build graph of connections (SIFT matches >= 35)
G = nx.Graph()
for p in image_paths:
    G.add_node(p.name)

match_details = defaultdict(dict)
n = len(image_paths)
for i in range(n):
    n1 = image_paths[i].name
    des1 = descriptors.get(n1)
    if des1 is None: continue
    for j in range(i + 1, n):
        n2 = image_paths[j].name
        des2 = descriptors.get(n2)
        if des2 is None: continue
        
        matches = bf.match(des1, des2)
        good = [m for m in matches if m.distance < 160]
        num_good = len(good)
        match_details[n1][n2] = num_good
        match_details[n2][n1] = num_good
        
        if num_good >= 35:
            G.add_edge(n1, n2, weight=num_good)

components = list(nx.connected_components(G))
components.sort(key=lambda c: len(c), reverse=True)

print(f"Initial inferred design groups count: {len(components)}")

# 3. Assign Design IDs and Colorway IDs
rows = []
design_audit_info = {}
singletons_list = []
multi_groups_list = []

for cid, comp in enumerate(components, 1):
    design_id = f"Design_{cid:03d}"
    comp_list = sorted(list(comp))
    is_single = (len(comp_list) == 1)
    
    if is_single:
        singletons_list.append((design_id, comp_list[0]))
    else:
        multi_groups_list.append((design_id, comp_list))
    
    # Audit info for group
    pairwise_info = []
    for idx1 in range(len(comp_list)):
        for idx2 in range(idx1 + 1, len(comp_list)):
            fn1, fn2 = comp_list[idx1], comp_list[idx2]
            score = match_details[fn1].get(fn2, 0)
            pairwise_info.append({'img1': fn1, 'img2': fn2, 'sift_matches': score})
            
    design_audit_info[design_id] = {
        'design_id': design_id,
        'image_count': len(comp_list),
        'is_singleton': is_single,
        'images': comp_list,
        'pairwise_sift_matches': pairwise_info
    }
    
    for c_idx, fn in enumerate(comp_list, 1):
        colorway_id = f"{design_id}_C{c_idx:02d}"
        meta = image_metadata[fn]
        rows.append({
            'image_path': meta['rel_path'],
            'design_id': design_id,
            'colorway_id': colorway_id,
            'is_singleton': is_single,
            'width': meta['width'],
            'height': meta['height'],
            'aspect_ratio': meta['aspect_ratio']
        })

# Sort rows by image_path
rows.sort(key=lambda x: x['image_path'])

# Write metadata.csv
csv_file = data_dir / "metadata.csv"
fieldnames = ['image_path', 'design_id', 'colorway_id', 'is_singleton', 'width', 'height', 'aspect_ratio']
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {csv_file} with {len(rows)} rows.")

# 4. Save dataset_summary.json
dimensions_list = [(r['width'], r['height']) for r in rows]
dim_counts = dict(Counter(dimensions_list))
dim_stats = {
    'unique_resolution_count': len(dim_counts),
    'most_common_resolutions': [{'width': k[0], 'height': k[1], 'count': v} for k, v in Counter(dimensions_list).most_common(5)],
    'min_width': min(r['width'] for r in rows),
    'max_width': max(r['width'] for r in rows),
    'min_height': min(r['height'] for r in rows),
    'max_height': max(r['height'] for r in rows),
    'aspect_ratios': dict(Counter(r['aspect_ratio'] for r in rows))
}

class_sizes = [len(comp) for comp in components]
summary_data = {
    'total_images': len(image_paths),
    'total_designs': len(components),
    'multi_image_designs': len(multi_groups_list),
    'singleton_designs': len(singletons_list),
    'images_in_multi_image_designs': sum(len(c) for _, c in multi_groups_list),
    'class_size_distribution': dict(Counter(class_sizes)),
    'image_dimension_statistics': dim_stats
}

summary_json = output_dir / "dataset_summary.json"
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(summary_data, f, indent=2)

print(f"Created {summary_json}.")

# Save audit details json
audit_json = output_dir / "group_audit.json"
with open(audit_json, 'w', encoding='utf-8') as f:
    json.dump(design_audit_info, f, indent=2)
print(f"Created {audit_json}.")

# 5. Generate Contact Sheets for Multi-Image Groups
# Split 28 multi-image groups across a few sheets (7 groups per sheet -> 4 sheets)
sheet_size = 7
for sheet_idx in range(0, len(multi_groups_list), sheet_size):
    sub_groups = multi_groups_list[sheet_idx : sheet_idx + sheet_size]
    max_imgs = max(len(imgs) for _, imgs in sub_groups)
    
    fig, axes = plt.subplots(nrows=len(sub_groups), ncols=max_imgs, figsize=(max_imgs * 3.2, len(sub_groups) * 3.2))
    fig.suptitle(f"Multi-Image Design Groups Contact Sheet (Sheet {sheet_idx//sheet_size + 1})", fontsize=16, fontweight='bold', y=0.995)
    
    if len(sub_groups) == 1:
        axes = np.array([axes])
    if max_imgs == 1:
        axes = np.array([[ax] for ax in axes])
        
    for r_idx, (did, comp_list) in enumerate(sub_groups):
        for c_idx in range(max_imgs):
            ax = axes[r_idx, c_idx]
            if c_idx < len(comp_list):
                fn = comp_list[c_idx]
                img_p = dataset_dir / "handloom_sarees" / fn
                im = Image.open(img_p)
                ax.imshow(im)
                colorway_id = f"{did}_C{c_idx+1:02d}"
                ax.set_title(f"{did}\n{colorway_id}\n{fn}", fontsize=9)
            ax.axis('off')
            
    plt.tight_layout()
    sheet_path = output_dir / f"contact_sheet_multi_groups_{sheet_idx//sheet_size + 1}.png"
    plt.savefig(sheet_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created {sheet_path}")

# 6. Generate Contact Sheet for Singletons (44 images: 6 rows x 8 cols)
cols = 8
rows_num = int(np.ceil(len(singletons_list) / cols))
fig, axes = plt.subplots(nrows=rows_num, ncols=cols, figsize=(cols * 2.8, rows_num * 2.8))
fig.suptitle("Singleton Design Classes (44 Unmatched Images)", fontsize=16, fontweight='bold', y=0.995)

axes_flat = axes.flatten()
for idx, (did, fn) in enumerate(singletons_list):
    ax = axes_flat[idx]
    img_p = dataset_dir / "handloom_sarees" / fn
    im = Image.open(img_p)
    ax.imshow(im)
    ax.set_title(f"{did}\n{fn}", fontsize=8)
    ax.axis('off')

for idx in range(len(singletons_list), len(axes_flat)):
    axes_flat[idx].axis('off')

plt.tight_layout()
singleton_sheet_path = output_dir / "contact_sheet_singletons.png"
plt.savefig(singleton_sheet_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Created {singleton_sheet_path}")
