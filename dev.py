import torch
import math

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

max_tokens = 16
maxpos = 4
attn_heads = 8
slopes = torch.Tensor(get_slopes(attn_heads))
#In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper). 
#If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
#This works because the softmax operation is invariant to translation, and our bias functions are always linear. 
alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
alibi = alibi.view(attn_heads, 1, maxpos)
alibi = alibi.repeat(max_tokens//maxpos, 1, 1)  # batch_size, 1, 1


def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y

get_relative_positions(4)

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

get_alibi_slope(8)

m = get_alibi_slope(8)
mask = torch.tril(torch.ones(1, 1, 4, 4))

(m * get_relative_positions(4)).unsqueeze(0)