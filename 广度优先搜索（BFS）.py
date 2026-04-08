import time
from collections import deque


def bfs_8_puzzle(start, goal):
    # 定义移动方向（上、下、左、右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 将状态转换为元组以便哈希
    start = tuple(start)
    goal = tuple(goal)

    # 初始化队列，每个元素包含当前状态、路径和空格位置
    queue = deque()
    queue.append((start, [], start.index(0)))

    # 记录已访问的状态
    visited = set()
    visited.add(start)

    # 统计扩展节点数
    expanded_nodes = 0

    # 开始计时
    start_time = time.time()

    while queue:
        current_state, path, zero_pos = queue.popleft()
        expanded_nodes += 1

        # 找到目标状态
        if current_state == goal:
            # 计算运行时间
            run_time = time.time() - start_time
            return path, run_time, expanded_nodes

        # 获取当前空格位置
        row, col = divmod(zero_pos, 3)

        # 生成所有可能的下一步状态
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # 计算新空格位置
                new_zero_pos = new_row * 3 + new_col
                # 生成新状态
                new_state = list(current_state)
                new_state[zero_pos], new_state[new_zero_pos] = new_state[new_zero_pos], new_state[zero_pos]
                new_state_tuple = tuple(new_state)

                # 如果新状态未被访问过
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    # 将新状态加入队列
                    queue.append((new_state_tuple, path + [new_state_tuple], new_zero_pos))

    # 如果没有找到解
    return None, time.time() - start_time, expanded_nodes


# 示例用法
if __name__ == "__main__":
    # 初始状态和目标状态（可根据需要修改）
    start_state = (2, 8, 4, 1, 6, 5, 7, 0, 3)
    goal_state = (1, 2, 3, 8, 0, 4, 7, 6, 5)

    # 调用BFS算法
    solution, run_time, expanded_nodes = bfs_8_puzzle(start_state, goal_state)

    if solution:
        # 输出解的步骤序列
        print("解的步骤序列：")
        for step in solution:
            print(step)

        # 输出统计信息
        print("\n算法运行时间：{:.6f}秒".format(run_time))
        print("扩展节点数：{}".format(expanded_nodes))
    else:
        print("未找到解！")