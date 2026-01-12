# visualize_two_images.py コマンド例

```bash
python visualize_two_images.py \
    /workspace/underwater/dust3r/test/5c34529873a8df509ae57b58/down_caustics/view_0010/color/frame_0010_caustics_0001.png \
    /workspace/underwater/dust3r/test/5c34529873a8df509ae57b58/down_caustics/view_0015/color/frame_0015_caustics_0001.png \
    --checkpoint /workspace/underwater/mast3r/output_mast3r_ver6/checkpoint-best.pth \
    --output_dir /workspace/underwater/mast3r/output_mast3r_ver6/visualizations \
    --size 512
```

### JPG画像を使用
```bash
python visualize_two_images.py \
    /workspace/underwater/GoPro/80/1.pot_normal/GOPR0325.JPG \
    /workspace/underwater/GoPro/80/1.pot_normal/GOPR0326.JPG
```

## 出力ファイル

- `depth_conf_img1.png` - 画像1の深度マップと信頼度マップ
- `depth_conf_img2.png` - 画像2の深度マップと信頼度マップ
- `matching_visualization.png` - マッチング結果の可視化
- `beta_values_img1.txt` - 画像1のβパラメータ統計（テキスト）
- `beta_values_img2.txt` - 画像2のβパラメータ統計（テキスト）
- `beta_maps_img1.png` - 画像1のβマップ可視化（RGBチャンネル別）
- `beta_maps_img2.png` - 画像2のβマップ可視化（RGBチャンネル別）
- `background_color_img1.png` - 画像1の背景色（B_inf）可視化
- `background_color_img2.png` - 画像2の背景色（B_inf）可視化
- `input_images.png` - 入力画像の並列表示
