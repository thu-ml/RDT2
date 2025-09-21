import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

np.bool = np.bool_
import random

import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms

class SplitHalfAffineAugmenter(iaa.Augmenter):
    def __init__(
        self, 
        name=None, 
        random_state=None
    ):
        super().__init__(name=name, random_state=random_state)
        # 分别为左右半边定义仿射变换，如果未指定则使用默认参数
        self.affine_left = self.affine_right = iaa.Affine(
            # 平移增强 (Panning Augmentation)
            translate_percent={
                "x": (-0.02, 0.02),  # 水平平移±2%
                "y": (-0.02, 0.02)  # 垂直平移±2%
                # "x": (-0.05, 0.05),  # 水平平移±5%
                # "y": (-0.05, 0.05)  # 垂直平移±5%
            },
            # 旋转增强 (Rotation Augmentation)
            rotate=(-2, 2),  # 随机旋转±2度
            # 其他几何变换参数
            scale=(0.98, 1.02),  # 随机缩放98%-102%
        )

    def _augment_images(self, images, random_state, parents, hooks):
        results = []
        for img in images:
            h, w = img.shape[:2]
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid:]
            # 分别对左右半边应用仿射变换
            left_aug = self.affine_left.augment_image(left)
            right_aug = self.affine_right.augment_image(right)
            # 拼接回来
            concat = np.concatenate([left_aug, right_aug], axis=1)
            # 防止拼接后宽度与原图不一致（偶数宽度没问题，奇数宽度补齐）
            # if concat.shape[1] != w:
            #     # 取多余或缺失的像素进行修正
            #     if concat.shape[1] > w:
            #         concat = concat[:, :w]
            #     else:
            #         pad = w - concat.shape[1]
            #         concat = np.pad(concat, ((0,0),(0,pad),(0,0)), mode='edge')
            results.append(concat)
        return np.array(results, dtype=images.dtype)

    def get_parameters(self):
        return [self.affine_left, self.affine_right]


# Define our sequence of augmentation steps that will be applied to every image
seq = iaa.Sequential(
    [
        # ================= 几何变换 (Geometric Transformations) =================
        iaa.Sometimes(
            0.5,  # 50%概率应用几何变换
            SplitHalfAffineAugmenter()
        ),

        # ================= 噪声增强 (Noise Augmentations) =================
        iaa.OneOf([  # 随机选择一种噪声类型
            iaa.AdditiveGaussianNoise(  # 高斯噪声
                loc=0,  # 均值0
                scale=(0.0, 0.02 * 255),  # 标准差范围: 0到5.1 (255的2%)
                per_channel=0.5  # 50%概率独立应用于每个通道
            ),
            iaa.AdditiveLaplaceNoise(  # 拉普拉斯噪声
                scale=(0.0, 0.02 * 255),  # 强度范围
                per_channel=0.5
            ),
            iaa.AdditivePoissonNoise(  # 泊松噪声
                lam=(0.0, 0.02 * 255),  # lambda参数范围
                per_channel=0.5
            )
        ]),

        # ================= 模糊增强 (Blur Augmentations) =================
        iaa.Sometimes(
            0.168,  # 70%概率应用模糊效果
            iaa.OneOf([  # 随机应用0或1种模糊效果
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.5)),  # 高斯模糊(模糊范围0-1.5像素)
                    iaa.AverageBlur(k=(3, 5)),  # 平均模糊(核大小3×3到5×5)
                    iaa.MedianBlur(k=(3, 5)),  # 中值模糊(核大小3×3到5×5)
                ]),
                iaa.MotionBlur(k=(3, 5)), 
            ]),
        ),
        iaa.Sometimes(
            0.4,
            # Higher values denote stronger compression
            iaa.JpegCompression(compression=(0, 50))
        ),
    ],
    random_order=True  # 以随机顺序执行所有增强操作
)


def simulate_strong_light(pil_img: Image.Image, strength='random'):
    """
    模拟强光曝光：强化亮部提升和暗部加深效果
    strength > 1 表示增强程度
    """
    if strength == 'random':
        import random
        strength = random.uniform(0.5, 1.0)  # 提高强度范围

    img = pil_img.convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # 归一化

    # 计算平均亮度
    gray_img = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    mean_intensity = np.mean(gray_img)

    # 非线性增强函数
    def nonlinear_enhance(x):
        # 计算每个像素偏离平均亮度的程度
        x = np.asarray(x, dtype=np.float32)

        # 以阈值为中心，计算偏差，范围[-0.5, 0.5]
        delta = x - mean_intensity

        # 使用 cosh 函数，cosh在两侧快速增长，中间最低，强化两端
        # 先归一化 delta 到 [-1, 1]
        norm_delta = delta / mean_intensity  # threshold=0.5时等价于 delta*2

        # 计算增强因子，减去最小值以保证中间附近输出约为0
        cosh_val = np.cosh(strength * norm_delta)
        cosh_val = cosh_val - np.cosh(0)

        # 将 cosh_val 归一化到 [0,1]
        cosh_norm = cosh_val / cosh_val.max()

        # 映射公式：
        # 中间区域变化小，cosh_norm 近0
        # 两端接近1，亮部和暗部调节大
        # 调整映射：x + (x - threshold) * cosh_norm
        mapped = x + (x - mean_intensity) * cosh_norm * 2.5

        # 限幅到0~1
        mapped = np.clip(mapped, 0.0, 1.0)
        return mapped

    enhanced_np = nonlinear_enhance(img_np)

    # 防止越界
    enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
    enhanced_img = Image.fromarray(enhanced_np)

    return enhanced_img

def image_corrupt(
    image: Image.Image,
    brightness=0.3, 
    contrast=0.4, 
    saturation=0.5, 
    hue=0.03
) -> Image.Image:
    # if random.random() < 0.2:
    #     image = simulate_strong_light(image, strength='random')
    
    image = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )(image)

    image_arr = np.array(image)
    image_arr = image_arr[None, ...]

    image_arr = seq(images=image_arr)

    image = Image.fromarray(image_arr[0])
    return image
