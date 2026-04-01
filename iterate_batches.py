from torch.utils.data import DataLoader
def iterate_batches(
    dataset,
    batch_size,
    shuffle
):
    """
    给定一个 PyTorch Dataset（数据集），返回一个按 `batch_size` 大小分批的可迭代对象（iterable）。
    完整遍历一次这个返回的可迭代对象，应该刚好等同于对该数据集进行了一个 epoch（完整的一轮训练/遍历）。

    参数 (Args):
        dataset: Dataset
            要从中提取数据批次（batches）的数据集。
        batch_size: int
            每个批次中需要包含的样本数量。
        shuffle: bool
            如果为 true，则在将样本打包成批次之前，先将它们的顺序随机打乱。

    返回值 (Returns):
        一个包含多个数据批次的可迭代对象，其中每个批次的大小为 `batch_size`。
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    