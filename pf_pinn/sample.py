from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt


def mesh_flat(*args):
    return [
        coord.reshape(-1, 1) for coord in jnp.meshgrid(*args)
    ]


def lhs_sampling(mins, maxs, num, key):
    dim = len(mins)
    u = (jnp.arange(0, num) + 0.5) / num  # 每个区间的中点
    
    keys = random.split(key, dim)
    result = jnp.zeros((num, dim))
    
    for i in range(dim):
        perm = random.permutation(keys[i], u)
        result = result.at[:, i].set(mins[i] + perm * (maxs[i] - mins[i]))
    
    return result

def shifted_grid(mins, maxs, nums, key):
    dim = len(mins)
    mins = jnp.array(mins)
    maxs = jnp.array(maxs)
    nums = jnp.array(nums)
    
    grids = []
    distances = (maxs - mins) / (nums - 1)
    
    keys = random.split(key, dim)
    shifts = jnp.array([random.uniform(keys[i], shape=(), 
                                      minval=-distances[i], 
                                      maxval=distances[i]) 
                        for i in range(dim)])
    
    # 创建带偏移的网格
    for i in range(dim):
        grid_i = jnp.linspace(mins[i], maxs[i], nums[i])[1:-1] + shifts[i]
        grids.append(grid_i)
    
    # 使用mesh_flat创建网格点
    return jnp.stack(mesh_flat(*grids), axis=-1).reshape(-1, dim)


# def lhs_sampling(mins, maxs, num):
#     lb = jnp.array(mins)
#     ub = jnp.array(maxs)
#     if not len(lb) == len(ub):
#         raise ValueError(f"mins and maxs should have the same length.")
#     ret = lhs(len(lb), int(num)) * (ub - lb) + lb
#     # return [ret[:, :-1], ret[:, -1:]]
#     return ret

# In Jax
# def shfted_grid(mins, maxs, num, key):
#     if not len(mins) == len(maxs) == len(num):
#         raise ValueError(f"mins, maxs, num should have the same length.")
    
#     each_col = [jnp.linspace(mins[i], maxs[i], num[i])[1:-1]
#                 for i in range(len(mins))]
#     distances = [(maxs[i] - mins[i]) / (num[i] - 1) for i in range(len(mins))]
#     shift = [random.uniform(key, (1,), minval=-distances[i], maxval=distances[i])
#              for i in range(len(distances))]
#     shift = jnp.concatenate(shift, axis=0)
#     each_col = [each_col[i] + shift[i] for i in range(len(each_col))]
#     return jnp.stack(mesh_flat(*each_col), axis=-1).reshape(-1, len(mins))

if __name__ == "__main__":
    mins = jnp.array([0, 0])
    maxs = jnp.array([1, 1])
    num = 100
    
    # 使用JAX的随机数生成器
    key = random.PRNGKey(42)
    key1, key2 = random.split(key)
    
    # 生成样本
    data = shifted_grid(mins, maxs, [20, 20], key1)
    data2 = shifted_grid(mins, maxs, [20, 20], key2)
    
    # 可视化
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Sample 1')
    plt.scatter(data2[:, 0], data2[:, 1], alpha=0.6, label='Sample 2')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Latin Hypercube Sampling (n={num})')
    plt.show()