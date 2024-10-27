import numpy as np

def randomTile_zero(images, tile_size, zero_ratio=0.2):
    """
    배치의 모든 이미지에 정해진 크기의 타일로 나눈 후, 랜덤한 타일을 선택하여 비율에 맞게 픽셀 값을 0으로 설정하는 함수.

    Args:
        images (torch.Tensor): 이미지 텐서 (B, C, H, W) 형태, 보통 배치, 채널, 높이, 너비.
        tile_size (tuple): 타일의 크기 (tile_height, tile_width)
        zero_ratio (float): 타일 중 몇 개를 0으로 설정할지 결정하는 비율 (0과 1 사이)

    Returns:
        modified_images (torch.Tensor): 랜덤 타일이 0으로 설정된 이미지 텐서
    """
    # 배치 내 모든 이미지를 처리
    batch_size, channels, height, width = images.shape
    tile_height, tile_width = tile_size

    # 이미지 내 타일의 개수 계산
    num_tiles_y = height // tile_height
    num_tiles_x = width // tile_width

    if num_tiles_y == 0 or num_tiles_x == 0:
        raise ValueError("Tile size is too large for the given image dimensions.")

    # 전체 타일 중 선택된 비율만큼 0으로 만들 타일의 개수 계산
    total_tiles = num_tiles_y * num_tiles_x
    num_tiles_to_zero = int(total_tiles * zero_ratio)

    # 원본 이미지를 복사하여 수정
    modified_images = images.clone()

    for i in range(batch_size):
        # 랜덤하게 타일을 선택하기 위해 타일 인덱스를 섞음
        all_tile_indices = [(y, x) for y in range(num_tiles_y) for x in range(num_tiles_x)]
        np.random.shuffle(all_tile_indices)
        selected_tiles = all_tile_indices[:num_tiles_to_zero]

        # 선택된 타일의 픽셀 값을 0으로 설정
        for tile_y, tile_x in selected_tiles:
            top = tile_y * tile_height
            left = tile_x * tile_width
            modified_images[i, :, top:top + tile_height, left:left + tile_width] = 0

    return modified_images

