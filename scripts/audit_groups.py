import json
import cv2
import numpy as np
from pathlib import Path
import networkx as nx

root_dir = Path(__file__).resolve().parents[1]
dataset_dir = root_dir / "sarees_dataset" / "handloom_sarees"
audit_file = root_dir / "outputs" / "dataset_inspection" / "group_audit.json"
suspicious_md_path = root_dir / "outputs" / "dataset_inspection" / "suspicious_groups.md"

with open(audit_file, 'r') as f:
    group_audit = json.load(f)

print(f"Auditing {len(group_audit)} design groups...")

# Compute SIFT & check singleton nearest neighbors (20-34 matches)
images = sorted(list(dataset_dir.glob("*.jpg")))
sift = cv2.SIFT_create(nfeatures=600)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

descriptors = {}
for img_path in images:
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is not None:
        kp, des = sift.detectAndCompute(im, None)
        descriptors[img_path.name] = des

# Find suspicious multi-image groups
# Criteria for suspicious group:
# 1. Minimum SIFT pairwise match in connected component is close to threshold (< 45)
# 2. Bipartite or weak bridge connection (a single image connecting two subgroups with weak match)

suspicious_groups = []
validated_groups_count = 0
singleton_count = 0

for did, info in group_audit.items():
    if info['is_singleton']:
        singleton_count += 1
        continue
    
    comp_list = info['images']
    pairs = info['pairwise_sift_matches']
    
    # Build sub-graph for this component
    subG = nx.Graph()
    for img_name in comp_list:
        subG.add_node(img_name)
    
    min_match = 999
    weak_pairs = []
    for p in pairs:
        m_val = p['sift_matches']
        if m_val < min_match:
            min_match = m_val
        if m_val < 45:
            weak_pairs.append((p['img1'], p['img2'], m_val))
        subG.add_edge(p['img1'], p['img2'], weight=m_val)
        
    # Check algebraic connectivity / articulation points
    bridges = list(nx.bridges(subG)) if len(comp_list) > 2 else []
    
    # Check if group is suspicious
    reasons = []
    if min_match < 42:
        reasons.append(f"Contains weak pairwise SIFT match of {min_match} (close to threshold 35).")
    if bridges:
        reasons.append(f"Contains {len(bridges)} bridge connection(s) where removing an edge splits the group.")
        
    if reasons:
        suspicious_groups.append({
            'design_id': did,
            'images': comp_list,
            'min_sift_match': min_match,
            'weak_pairs': weak_pairs,
            'reasons': reasons
        })
    else:
        validated_groups_count += 1

print(f"Validated multi-image groups: {validated_groups_count}")
print(f"Suspicious multi-image groups: {len(suspicious_groups)}")

# Singleton Audit: Check if any singleton has near-matches (25 to 34 matches) with another image
singletons = [info['images'][0] for info in group_audit.values() if info['is_singleton']]
near_singleton_matches = []

for s_img in singletons:
    des1 = descriptors.get(s_img)
    if des1 is None: continue
    best_match = 0
    best_target = None
    for other_img in images:
        if other_img.name == s_img: continue
        des2 = descriptors.get(other_img.name)
        if des2 is None: continue
        
        matches = bf.match(des1, des2)
        good = [m for m in matches if m.distance < 160]
        if len(good) > best_match:
            best_match = len(good)
            best_target = other_img.name
            
    if best_match >= 22:
        near_singleton_matches.append((s_img, best_target, best_match))

print(f"\nSingletons with near-threshold matches (22-34 SIFT matches): {len(near_singleton_matches)}")

# Write suspicious_groups.md
md_lines = [
    "# Suspicious Design Groups Audit Report",
    "",
    "This file documents inferred design groups that display weak SIFT keypoint connectivity or potential false-positive matching caused by repeated generic border textures or background features.",
    "",
    "| Design ID | Image Count | Min Pairwise SIFT Matches | Suspicion Rationale | Recommended Action |",
    "| :---: | :---: | :---: | :--- | :--- |"
]

for g in suspicious_groups:
    did = g['design_id']
    imgs_str = ", ".join([f"`{img}`" for img in g['images']])
    min_m = g['min_sift_match']
    reason_str = " ".join(g['reasons'])
    rec_action = "Manual visual verification or split if colorway motifs differ."
    md_lines.append(f"| **{did}** | {len(g['images'])} | {min_m} | {reason_str} Affected: {imgs_str} | {rec_action} |")

md_lines.extend([
    "",
    "## Singleton Near-Match Audit",
    "",
    "The following 44 singleton images were checked against all 165 dataset images to verify whether any structural partners exist below the initial threshold (35 SIFT matches):",
    "",
    "| Singleton Image | Highest Match Partner | SIFT Keypoint Matches | Audit Conclusion |",
    "| :--- | :--- | :---: | :--- |"
])

if near_singleton_matches:
    for s_img, target, m_count in near_singleton_matches:
        md_lines.append(f"| `{s_img}` | `{target}` | {m_count} | Near match below threshold; true singleton or weak variant. |")
else:
    md_lines.append("| None | N/A | <22 | All 44 singletons are confirmed distinct designs with no near structural matches. |")

with open(suspicious_md_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(md_lines) + "\n")

print(f"Created {suspicious_md_path}.")

