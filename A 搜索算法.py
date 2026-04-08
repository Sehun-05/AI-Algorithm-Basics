import time


class Astar:
    def solvePuzzle(self, init, targ):
        start_time = time.time()  # 记录开始时间
        open = [(init, self.calcDistH(init, targ))]
        closed = {}
        dict_depth = {init: 0}
        dict_link = {}
        dict_dirs = {'u': [-1, 0], 'd': [1, 0], 'l': [0, -1], 'r': [0, 1]}
        dirs = ['l', 'r', 'u', 'd']
        expanded_nodes = 0  # 扩展节点计数器
        found = False

        while open:
            open.sort(key=lambda x: x[1])
            # 清理closed表中的重复节点
            while open and open[0][0] in closed:
                open.pop(0)
            if not open:
                break

            current_node = open[0]
            open.pop(0)

            if current_node[0] == targ:
                found = True
                break

            closed[current_node[0]] = current_node[1]
            expanded_nodes += 1  # 节点扩展计数
            cur_index = current_node[0].index('0')

            # 生成所有可能的移动方向
            for i in range(4):
                x = cur_index // 3 + dict_dirs[dirs[i]][0]
                y = cur_index % 3 + dict_dirs[dirs[i]][1]
                if 0 <= x < 3 and 0 <= y < 3:
                    next_index = x * 3 + y
                    next_state = self.moveMap(current_node[0], cur_index, next_index)
                    depth = dict_depth[current_node[0]] + 1

                    # 更新节点信息
                    if next_state not in dict_depth or depth < dict_depth[next_state]:
                        dict_depth[next_state] = depth
                        dict_link[next_state] = current_node[0]

                        # 如果节点在closed表中，需要移出并重新加入open表
                        if next_state in closed:
                            del closed[next_state]

                        # 计算新的f值并加入open表
                        new_f = depth + self.calcDistH(next_state, targ)
                        open.append((next_state, new_f))

        # 生成路径信息
        end_time = time.time()
        runtime = end_time - start_time
        states = []
        moves = []

        if found:
            # 回溯生成状态序列和移动路径
            current = targ
            while current != init:
                states.append(current)
                move = current.index('0') - dict_link[current].index('0')

                if move == -1:
                    moves.append('l')
                elif move == 1:
                    moves.append('r')
                elif move == -3:
                    moves.append('u')
                else:
                    moves.append('d')

                current = dict_link[current]

            states.append(init)
            states.reverse()
            moves.reverse()

            # 打印统计信息和路径
            print("=" * 40)
            print(f"算法运行时间: {runtime:.4f}秒")
            print(f"扩展节点数量: {expanded_nodes}")
            print("状态转移序列:")
            for state in states:
                print(state, end=" -> ")
            print("\b\b\b")  # 删除最后的箭头
            print("\n移动路径序列:", "".join(moves))
            print("=" * 40)
            return "".join(moves)
        else:
            print("无解")
            return ""

    def calcDistH(self, src_map, dest_map):
        cost = 0
        for i in range(9):
            if src_map[i] != '0':
                num = int(src_map[i])
                cost += abs(num // 3 - i // 3) + abs(num % 3 - i % 3)
        return cost

    def moveMap(self, cur_map, i, j):
        cur_list = list(cur_map)
        cur_list[i], cur_list[j] = cur_list[j], cur_list[i]
        return "".join(cur_list)


# 测试用例
if __name__ == "__main__":
    solver = Astar()
    solution = solver.solvePuzzle("724506831", "012345678")
    print("最终解路径:", solution)