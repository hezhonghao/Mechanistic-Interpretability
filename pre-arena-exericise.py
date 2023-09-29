import math
from einops import rearrange, repeat, reduce
import torch as t
from fancy_einsum import einsum


# Assertion that guarantees both shape and value match
def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


# Assertion that guarantees shape matches
def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


def rearrange_1() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    return rearrange(t.arange(3,9), '(c d) -> c d', c=3,d=2) # notice the spelling diff between rearrange and t.arange 

print(rearrange_1())
expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


def rearrange_2() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return rearrange(t.arange(1,7), '(c d) -> c d', c=2,d=3) #t.arange includes the former but not the later


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


def rearrange_3() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    """
    return rearrange(t.arange(1,7), 'a -> 1 a 1') # Einops dim can include both/either letter and/or number.


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))


def temperatures_average(temps: t.Tensor) -> t.Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return reduce(temps, '(a b) -> a', b=7, reduction="mean")   # output dim（3，） # For reduce you will need keyword argument "reduction="


temps = t.Tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83])
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)


def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    t_mean = temperatures_average(temps)  
    t_mean_21 = t.repeat_interleave(t_mean, 7) # This gets [1,1,1,1,1,1,1,2,2,2...]; you may also try repeat(input, 'm -> (m 7)')
    # This actually repeats tensor entries not in a desired way ([1,2,3,1,2,3,...]): t_mean_21 = repeat(t_mean, 'a -> (c a)', c=7)   # (21,) Notice how do you handle dim change in einops  This is a tensor of 21 entries, each a mean fo seven days. Only three different values.
    print('t_mean21:', t_mean_21, 'temps - t_mean_21:', temps - t_mean_21)
    return temps - t_mean_21 # both (21,)
 

expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)




def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """
    avg = repeat(temperatures_average(temps), "w -> (w 7)")
    std = repeat((reduce(temps, "(w 7) -> w", t.std)), "h -> (h 7)")  # Surprisingly you may leave "t.std" as a method of reduce; 
    # when you "repeat", "h -> (h 7)" and "h -> (7 h)" gives you diff repetitions. "h -> (h 7)" gives you 1 1 1 1 1 1 1,2...; "h -> (7 h)" gives you (1 2 3, 1 2 3...). I guess it's just convention 
    return (temps - avg) / std
    


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)


def batched_dot_product_nd(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Return the batched dot product of a and b, where the first dimension is the batch dimension.

    That is, out[i] = dot(a[i], b[i]) for i in 0..len(a).
    a and b can have any number of dimensions greater than 1.

    a: shape (b, i_1, i_2, ..., i_n)
    b: shape (b, i_1, i_2, ..., i_n)

    Returns: shape (b, )

    Use torch.einsum. You can use the ellipsis "..." in the einsum formula to represent an arbitrary number of dimensions.
    """
    assert a.shape == b.shape
    return t.einsum('b...i,b...i->b', a,b) # Batched dot product returns a vector of scalars 
    # Please notice the batched dot product with einsum: there is an "i" in the end.

actual = batched_dot_product_nd(t.tensor([[1, 1, 0], [0, 0, 1]]), t.tensor([[1, 1, 0], [1, 1, 0]]))
expected = t.tensor([2, 0])
assert_all_equal(actual, expected)
actual2 = batched_dot_product_nd(t.arange(12).reshape((3, 2, 2)), t.arange(12).reshape((3, 2, 2)))
expected2 = t.tensor([14, 126, 366])
assert_all_equal(actual2, expected2)


def identity_matrix(n: int) -> t.Tensor:
    """Return the identity matrix of size nxn.

    Don't use torch.eye or similar.

    Hint: you can do it with arange, rearrange, and ==.
    Bonus: find a different way to do it.
    """
    assert n >= 0
    if n == 0:
        return t.zeros((0,0))  # The rearrange function cannot handle sceanrio in which we jhave a empty tensor (t.arange(0,0))
    a = rearrange(t.arange(0,n*n), '(b d) -> b d', b=n, d=n)  # In the einops rearrange function, each dimension should have a unique identifier and the spaces are important as they separate dimensions.
    row, column = a.shape
    identity = t.tensor([[1 if i ==j else 0 for j in range(row)] for i in range(column)])
    print('identity shape:',identity.shape)
    print('identity:', identity)
    return identity

    # different way  
    a = t.arange(0,n)
    b = rearrange(a, 'n -> n ()')  # smart way of converting an 1D tensor to 2D one.
    c = rearrange(a, 'n -> () n')
    identity = (b==c).float()  # from boolean type to float 


assert_all_equal(identity_matrix(3), t.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
assert_all_equal(identity_matrix(0), t.zeros((0, 0))) 


def sample_distribution(probs: t.Tensor, n: int) -> t.Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    I am supposed to return a tensor of events occured, under the probs identified by "probs", but the crux is "what are the events to occur is not clear to me".
    I feel that i don't know what to be returned. 

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples
    Return n random samples from probs, where probs is a normalized probability distribution. # So for each entry of the output, do you want a prob number, or an integer? --> yes an integer. But where the integer from?
    Return: shape (n,) where out[i] is an integer indicating which event was sampled. # The integer should be from [0,n) Or should be n number of probs?

    Use torch.rand and torch.cumsum to do this without any explicit loops.

    Note: if you think your solution is correct but the test is failing, try increasing the value of n.
    """
    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()

    return (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)
    # I don't understand this at all.
    


n = 10000000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs, rtol=0.001, atol=0.001)


def classifier_accuracy(scores: t.Tensor, true_classes: t.Tensor) -> t.Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use torch.argmax.
    """
    predicted = t.argmax(scores, dim=1) # (batch,) Returned index of the max values among all (of flattened tensor)
    accurate = 0
    batch = true_classes.shape[0]
    accurate += sum([1 if predicted[i] == true_classes[i] else 0 for i in range(batch)])  # if you want to use for loop you have to have a list comprehension to include every element. Thus, you transform the list into an integer bc you want an integer.
    accuracy = accurate/batch 
    return accuracy 
    assert true_classes.max() < scores.shape[1]



scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected


def total_price_indexing(prices: t.Tensor, items: t.Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item. (NEP: I think this refers to all available items, not necessarily items purchased )
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).  (NEP: n items purchased )

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    assert items.max() < prices.shape[0]
    return t.sum(prices[items]) # Simple and beautiful
    
    # In essence I think this is to index items from another list and to put on a new list 



prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0


def gather_2d(matrix: t.Tensor, indexes: t.Tensor) -> t.Tensor:
    """Perform a gather operation along the second dimension. --> NEP: Gather items along certain dim

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). 
    out[i][j] = matrix[i][indexes[i][j]]  For this index, notice index[i][j] return a single value, as a further index that identifies which column of matrix you point to.

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    "TODO: YOUR CODE HERE"
    out = matrix.gather(1, indexes)
    "TODO: YOUR CODE HERE"
    return out  


matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)
indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes), expected)
# assert out.shape == indexes.shape
# 其他assertion还写什么

def total_price_gather(prices: t.Tensor, items: t.Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather.

    Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item. (NEP: I think this refers to all available items, not necessarily items purchased )
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).  (NEP: n items purchased )
    """ 
    
    assert items.max() < prices.shape[0]

    return t.sum(t.gather(prices, 0, items)) # items serves as index while prices is the input  t.gather returns a tensor of all the prices of items you purchased, so you will need a total price.

    # t.gather is used when you want to gather items along one dim.

prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0


def integer_array_indexing(matrix: t.Tensor, coords: t.Tensor) -> t.Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n) 
    coords: shape (batch, n) 
    Return: (batch, )
    """
    coords_tuple = tuple(coords.t()) # We convert coords to a tuple, such that each tensor in it in charge of indexing one dim.
    return matrix[coords_tuple] 


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))
mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))


def batched_logsumexp(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way. # What is a numerically stable way?

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.
    # NEP: I am not sure about the purpose of this function tho. 
    # Oh I see it's some computational stability reason, as suggested (you migh run into log0 or log(infinite))
    # Oh i see I mostly get the intuition and will do the whole formula next time.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

    """
    batch, _ = matrix.shape
    out = t.rand(batch) # initialize out as a (batch,) tensor
    for i in range(batch):
        a = t.max(matrix[i])
        out[i] = a + t.log(t.sum(t.exp(matrix[i]-a))) # A broadcast here "matrix[i]-a"
    return out 
    # can you do this with no for loop? TODO


matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)
matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)


def batched_softmax(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1. # This i is the index for the batch, for each batch sum to 1  
    """
    # This should return a 2-D tensor where each row is a mormalized vector 
    
    exp_matrix = t.exp(matrix) # this is vectorized computation, and therefore you won't worry about "compute softmax(row)"
    sum_exp_matrix = t.sum(exp_matrix, dim=1, keepdim=True) # sum over columns (--> columns to be reduced)
    # From here you understand more of vectorization. i j index seem messy, but it seems you only need to pay attention to where you need to sum them.

    return  exp_matrix / sum_exp_matrix


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))
matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))


def batched_logsoftmax(matrix: t.Tensor) -> t.Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """

    a, _ = t.max(matrix, dim=1, keepdim=True) # max over each row (columns to be reduced); we only care about max_values not max_indices
    exp_matrix = t.exp(matrix-a) # a broadcast, each entries you subtract max of that row. # according to my calculation by hand, we don't need to add this a back.
    print('this shape:', exp_matrix.shape)
    sum_exp_matrix = t.sum(exp_matrix, dim=1, keepdim=True) # sum over columns
    log_sum_exp_matrix = t.log(exp_matrix / sum_exp_matrix) # the logsoftmax formula
    print('logsoftmax:', log_sum_exp_matrix) # test
    return  log_sum_exp_matrix





matrix = t.arange(1, 6).view((1, 5)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 6).view((1, 5)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-4.4519, -3.4519, -2.4519, -1.4519, -0.4519]])
assert_all_close(actual, expected)


def batched_cross_entropy_loss(logits: t.Tensor, true_labels: t.Tensor) -> t.Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    log_prob = batched_logsoftmax(logits) 
    true_preds = t.gather(input = -log_prob, dim=1,index=true_labels.unsqueeze(-1)) # performing cross-entropy_loss; this is the simplified version for 1 data point and C classes (multiple data points you just vectorize it)
    out = true_preds.squeeze(-1)  # you first add one dim (to match inpiut dim and index dim) then cut it.
    return out # returning cross entropy with shape (batch, )


logits = t.tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
true_labels = t.tensor([2, 0, 0])
expected = t.tensor([0.0, math.log(3), float("inf")])
actual = batched_cross_entropy_loss(logits, true_labels)
assert_all_close(actual, expected)


def collect_rows(matrix: t.Tensor, row_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    assert row_indexes.max() < matrix.shape[0]
    return matrix[row_indexes, :] 


matrix = t.arange(15).view((5, 3))
row_indexes = t.tensor([0, 2, 1, 0])
actual = collect_rows(matrix, row_indexes)
expected = t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
assert_all_equal(actual, expected)


def collect_columns(matrix: t.Tensor, column_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    """
    assert column_indexes.max() < matrix.shape[1]
    return matrix[:, column_indexes]
    # Alternatively I think this might work
    # return [matrix[:, i] for i in column_indexes] Although this is a list of vectors rather than a matrix. Besides tyhis might be more computatioanlly demanding


matrix = t.arange(15).view((5, 3))
column_indexes = t.tensor([0, 2, 1, 0])
actual = collect_columns(matrix, column_indexes)
expected = t.tensor([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]])
assert_all_equal(actual, expected)