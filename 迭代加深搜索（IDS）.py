import time
from collections import deque

def ids_8_puzzle(start, goal):
    # 定义移动方向（上、下、左、右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 将状态转换为元组以便哈希
    start = tuple(start)
    goal = tuple(goal)

    # 统计扩展节点数
    expanded_nodes = 0

    # 开始计时
    start_time = time.time()

    # 迭代加深搜索主循环
    depth_limit = 0
    solution = None

    while True:
        # 使用DFS进行受限深度搜索
        stack = [(start, [], start.index(0))]
        visited = set()  # 记录当前深度限制下的已访问状态
        found = False

        while stack:
            current_state, path, zero_pos = stack.pop()

            # 检查是否达到当前深度限制
            if len(path) > depth_limit:
                continue

            expanded_nodes += 1

            # 找到目标状态
            if current_state == goal:
                found = True
                solution = path + [current_state]
                break

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

                    # 检查是否已访问（当前深度限制下）
                    if new_state_tuple not in visited:
                        visited.add(new_state_tuple)
                        # 将新状态加入栈
                        stack.append((new_state_tuple, path + [current_state], new_zero_pos))

        # 如果找到解或深度限制超过可能最大深度则退出
        if found or depth_limit > 100:  # 八数码问题最大可能深度为 20
            break
        depth_limit += 1

    # 计算运行时间
    run_time = time.time() - start_time

    return solution, run_time, expanded_nodes


# 示例用法
if __name__ == "__main__":
    # 可解的示例（逆序数均为偶数）
    start_state = (3, 1, 4, 8, 5, 6, 7, 0, 2)
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # 逆序数为0（偶数）

    # 调用IDS算法
    solution, run_time, expanded_nodes = ids_8_puzzle(start_state, goal_state)

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