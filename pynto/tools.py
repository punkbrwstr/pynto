from typing import Union
import numpy as np
import numpy.typing as npt

_ary64 = npt.NDArray[np.float64]

def expanding_mean(x: np.ndarray, _: int) -> np.ndarray:
    nan_mask: npt.NDArray[np.bool] = np.isnan(x)
    cumsum: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x))
    count: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return np.where(nan_mask, np.nan, cumsum) / count

def expanding_var(x: np.ndarray, _: int) -> np.ndarray:
    nan_mask: npt.NDArray[np.bool] = np.isnan(x)
    cumsum: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x))
    cumsumOfSquares: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x * x))
    count: _ary64 = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return (cumsumOfSquares - cumsum * cumsum / count) / (count - 1)

def expanding_std(x: np.ndarray, _: int) -> np.ndarray:
    nan_mask = np.isnan(x)
    cumsum = np.add.accumulate(np.where(nan_mask, 0, x))
    cumsumOfSquares = np.add.accumulate(np.where(nan_mask, 0, x * x))
    count = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return np.sqrt((cumsumOfSquares - cumsum * cumsum / count) / (count - 1))

def ewma(data: np.ndarray, alpha: float) -> np.ndarray:

    dtype = np.float64 if data.dtype != np.float32 else np.float32
    row_size = _get_max_row_size(data, alpha)
    out = np.empty_like(data, dtype=dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return _ewma_vectorized(data, alpha, offset=0, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order='C')


    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    _ewma_vectorized_2d(data_main_view, alpha, axis=1,
                    offset=np.array([0]),out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        _ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                            out=out[-trailing_n:])
    return out

def _get_max_row_size(data: np.ndarray, alpha: float):
    assert 0. <= alpha < 1.
    dtype = np.float64 if data.dtype != np.float32 else np.float32
    epsilon = np.finfo(dtype).tiny
    return int(np.log(epsilon)/np.log(1-alpha)) + 1

def _ewma_vectorized(data: np.ndarray, alpha: float,
            offset: int, out: np.ndarray) -> np.ndarray:

    dtype = np.float64 if data.dtype != np.float32 else np.float32

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order='C')

    assert out.shape == data.shape
    assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    alpha_ary: _ary64 = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha_ary, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha_ary * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset_ary = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset_ary * scaling_factors[1:]

    return out

def _ewma_vectorized_2d(data: np.ndarray, alpha: float, axis: int,
                    offset: np.ndarray, out: np.ndarray):
    dtype = np.float64 if data.dtype != np.float32 else np.float32
    assert data.ndim <= 2
    assert out.shape == data.shape
    assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        return _ewma_vectorized(data, alpha, offset[0], out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    alpha_ary:_ary64 = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha_ary, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha_ary * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out


def corrcoeff_1d(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1,keepdims=1)
    B_mB = B - B.mean(-1,keepdims=1)
    
    # Sum of squares
    ssA = np.einsum('i,i->',A_mA, A_mA)
    ssB = np.einsum('i,i->',B_mB, B_mB)
    
    # Finally get corr coeff
    return np.einsum('i,i->',A_mA,B_mB)/np.sqrt(ssA*ssB)

# https://stackoverflow.com/a/40085052/ @ Divakar
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

# https://stackoverflow.com/a/41703623/ @Divakar
def corr2_coeff_rowwise(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1,keepdims=1)
    B_mB = B - B.mean(-1,keepdims=1)

    # Sum of squares across rows
    ssA = np.einsum('ij,ij->i',A_mA, A_mA)
    ssB = np.einsum('ij,ij->i',B_mB, B_mB)

    # Finally get corr coeff
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)

def nancorrcoeff_1d(A: np.ndarray,B: np.ndarray):
    # Get combined mask
    comb_mask: npt.NDArray[np.bool] = ~(np.isnan(A) & ~np.isnan(B))
    count = comb_mask.sum()

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.nansum(A * comb_mask,-1,keepdims=True)/count
    B_mB = B - np.nansum(B * comb_mask,-1,keepdims=True)/count

    # Replace NaNs with zeros, so that later summations could be computed    
    A_mA[~comb_mask] = 0
    B_mB[~comb_mask] = 0

    ssA = np.inner(A_mA,A_mA)
    ssB = np.inner(B_mB,B_mB)

    # Finally get corr coeff
    return np.inner(A_mA,B_mB)/np.sqrt(ssA*ssB)

def nancorrcoeff_rowwise(A,B):
    # Input : Two 2D arrays of same shapes (mxn). Output : One 1D array  (m,)
    # Get combined mask
    comb_mask = ~(np.isnan(A) & ~np.isnan(B))
    count = comb_mask.sum(axis=-1,keepdims=1)

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.nansum(A * comb_mask,-1,keepdims=True)/count
    B_mB = B - np.nansum(B * comb_mask,-1,keepdims=True)/count

    # Replace NaNs with zeros, so that later summations could be computed    
    A_mA[~comb_mask] = 0
    B_mB[~comb_mask] = 0

    # Sum of squares across rows
    ssA = np.einsum('ij,ij->i',A_mA, A_mA)
    ssB = np.einsum('ij,ij->i',B_mB, B_mB)

    # Finally get corr coeff
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)

import random

class SkipListNode:
    """Node in a skip list with multiple forward references."""
    def __init__(self, element=None, score=None, level=0):
        self.element = element
        self.score = score
        # Array of forward references for each level
        self.forward = [None] * (level + 1)
        
class SkipList:
    """Skip list implementation for Redis sorted set."""
    MAX_LEVEL = 32  # Same as Redis default
    P_FACTOR = 0.25  # Probability factor for level assignment
    
    def __init__(self):
        # Create header node (doesn't store an actual element)
        self.header = SkipListNode(level=self.MAX_LEVEL)
        self.level = 0  # Current maximum level in the skip list
        self.length = 0  # Number of elements in the skip list
        # Map for O(1) element lookup and duplicate prevention
        self.dict = {}
        
    def _random_level(self):
        """Randomly determine the level for a new node."""
        level = 0
        while random.random() < self.P_FACTOR and level < self.MAX_LEVEL:
            level += 1
        return level
        
    def insert(self, element, score):
        """Insert an element with the given score into the skip list."""
        # Check if element already exists
        if element in self.dict:
            old_score = self.dict[element].score
            # If score is the same, nothing to do
            if old_score == score:
                return 0
            # Otherwise, remove it first and then re-insert with new score
            self.delete(element)
        
        # Array to keep track of update points at each level
        update = [None] * (self.MAX_LEVEL + 1)
        # Current node for traversal
        x = self.header
        
        # Find the position to insert the new node at each level
        for i in range(self.level, -1, -1):
            while (x.forward[i] and 
                   (x.forward[i].score < score or 
                    (x.forward[i].score == score and x.forward[i].element < element))):
                x = x.forward[i]
            update[i] = x
        
        # Generate a random level for the new node
        level = self._random_level()
        
        # Update the skip list's level if necessary
        if level > self.level:
            for i in range(self.level + 1, level + 1):
                update[i] = self.header
            self.level = level
        
        # Create the new node
        x = SkipListNode(element, score, level)
        
        # Update the forward references
        for i in range(level + 1):
            x.forward[i] = update[i].forward[i]
            update[i].forward[i] = x
        
        # Add to dictionary for O(1) lookup
        self.dict[element] = x
        self.length += 1
        return 1
        
    def delete(self, element):
        """Remove an element from the skip list."""
        if element not in self.dict:
            return 0
            
        # Array to keep track of update points at each level
        update = [None] * (self.MAX_LEVEL + 1)
        # Current node for traversal
        x = self.header
        
        # Find the node to delete at each level
        for i in range(self.level, -1, -1):
            while (x.forward[i] and 
                   (x.forward[i].score < self.dict[element].score or 
                    (x.forward[i].score == self.dict[element].score and 
                     x.forward[i].element < element))):
                x = x.forward[i]
            update[i] = x
        
        # Get the node to delete
        x = x.forward[0]
        
        # Make sure we have the right element
        if x and x.element == element:
            # Update forward references to bypass this node
            for i in range(self.level + 1):
                if update[i].forward[i] != x:
                    break
                update[i].forward[i] = x.forward[i]
            
            # Update the max level if necessary
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
            
            # Remove from dictionary
            del self.dict[element]
            self.length -= 1
            return 1
        
        return 0
    
    def score(self, element):
        """Get the score of an element."""
        node = self.dict.get(element)
        return node.score if node else None
    
    def rank(self, element, reverse=False):
        """Return the rank of the element in the sorted set."""
        if element not in self.dict:
            return None
            
        target_score = self.dict[element].score
        target_element = element
        rank = 0
        
        # Skip list traversal to count elements
        x = self.header
        for i in range(self.level, -1, -1):
            while (x.forward[i] and 
                   (x.forward[i].score < target_score or 
                    (x.forward[i].score == target_score and 
                     x.forward[i].element < target_element))):
                rank += 1
                x = x.forward[i]
        
        # Handle reverse order if requested
        if reverse:
            rank = self.length - rank - 1
            
        return rank
    
    def range_by_rank(self, start, end=None, reverse=False):
        """Return elements within the given rank range."""
        if end is None:
            end = -1
            
        if end < 0:
            end = max(0, self.length + end)
            
        if start < 0:
            start = max(0, self.length + start)
            
        if start > end or start >= self.length:
            return []
            
        end = min(end, self.length - 1)
        
        # Convert ranks for reversed order if needed
        if reverse:
            start, end = self.length - end - 1, self.length - start - 1
        
        result = []
        x = self.header
        traversed = 0
        
        # Find the start position using the skip list's levels
        for i in range(self.level, -1, -1):
            while x.forward[i] and traversed + 1 <= start:
                traversed += 1
                x = x.forward[i]
        
        # Collect elements from start to end
        x = x.forward[0]  # Move to the first element in range
        while x and traversed <= end:
            result.append((x.element, x.score))
            traversed += 1
            x = x.forward[0]
            
        # Reverse the result if needed
        if reverse:
            result.reverse()
            
        return result
    
    def range_by_score(self, min_score, max_score, start=0, count=None, reverse=False):
        """Return elements with scores between min_score and max_score."""
        result = []
        
        # Skip list traversal
        x = self.header
        
        # Start from the top level and work down
        for i in range(self.level, -1, -1):
            # Skip forward to min_score
            while x.forward[i] and x.forward[i].score < min_score:
                x = x.forward[i]
        
        # Now x.forward[0] is >= min_score or null
        x = x.forward[0]
        
        # Skip 'start' elements
        skipped = 0
        while x and skipped < start and x.score <= max_score:
            skipped += 1
            x = x.forward[0]
        
        # Collect elements within the score range
        collected = 0
        while x and x.score <= max_score:
            result.append((x.element, x.score))
            collected += 1
            
            if count is not None and collected >= count:
                break
                
            x = x.forward[0]
        
        # Reverse if needed
        if reverse:
            result.reverse()
            
        return result
    
    def range_by_lex(self, min_lex, max_lex, start=0, count=None, reverse=False):
        """Return elements lexicographically ordered between min_lex and max_lex."""
        # Parse range boundaries
        min_value, max_value, inclusive_min, inclusive_max = self._parse_lex_range(min_lex, max_lex)
        
        # Get all elements with the same score
        # For a true lexicographical comparison, we consider elements with score 0
        lex_score = 0
        
        # Skip list traversal
        x = self.header
        result = []
        
        # Skip to the first element >= min_value
        for i in range(self.level, -1, -1):
            while (x.forward[i] and 
                   (x.forward[i].score < lex_score or 
                    (x.forward[i].score == lex_score and 
                     ((inclusive_min and x.forward[i].element < min_value) or 
                      (not inclusive_min and x.forward[i].element <= min_value))))):
                x = x.forward[i]
        
        # Now x.forward[0] is the first potential element in range
        x = x.forward[0]
        
        # Skip 'start' elements
        skipped = 0
        while (x and skipped < start and x.score == lex_score and
               ((inclusive_max and x.element <= max_value) or 
                (not inclusive_max and x.element < max_value))):
            skipped += 1
            x = x.forward[0]
        
        # Collect elements within the lexicographical range
        collected = 0
        while (x and x.score == lex_score and
               ((inclusive_max and x.element <= max_value) or 
                (not inclusive_max and x.element < max_value))):
            result.append(x.element)
            collected += 1
            
            if count is not None and collected >= count:
                break
                
            x = x.forward[0]
        
        # Reverse if needed
        if reverse:
            result.reverse()
            
        return result
    
    def _parse_lex_range(self, min_lex, max_lex):
        """Parse lexicographical range boundaries like Redis."""
        # Define boundary types
        inclusive_min = min_lex[0] != "("
        inclusive_max = max_lex[0] != "("
        
        # Extract the actual string values
        if min_lex[0] in "([":
            min_value = min_lex[1:]
        else:
            min_value = min_lex
            inclusive_min = True
            
        if max_lex[0] in "([":
            max_value = max_lex[1:]
        else:
            max_value = max_lex
            inclusive_max = True
            
        # Handle Redis special values
        if min_lex == "-":
            min_value = ""
            inclusive_min = True
        if max_lex == "+":
            max_value = "\uffff"  # A high Unicode value that will be greater than any string
            inclusive_max = True
            
        return min_value, max_value, inclusive_min, inclusive_max

class SortedSet:
    """
    Redis-like Sorted Set implementation using a Skip List.
    Provides both score-based ordering and lexicographical queries.
    """
    def __init__(self):
        self.skiplist = SkipList()
        # For lexicographical ordering without scores, we use a constant score
        self.LEX_SCORE = 0
        
    def add(self, element, score=None):
        """
        Add an element with the given score, or use LEX_SCORE for lexicographical ordering.
        Returns 1 if the element is new, 0 if the score was updated.
        """
        if score is None:
            score = self.LEX_SCORE
        return self.skiplist.insert(element, score)
        
    def remove(self, element):
        """Remove an element from the sorted set."""
        return self.skiplist.delete(element)
        
    def score(self, element):
        """Get the score of an element."""
        return self.skiplist.score(element)
        
    def card(self):
        """Return the cardinality (number of elements) of the sorted set."""
        return self.skiplist.length
        
    def rank(self, element, reverse=False):
        """Return the rank of the element in the sorted set."""
        return self.skiplist.rank(element, reverse)
        
    def range_by_rank(self, start, end=None, reverse=False):
        """Return elements within the given rank range."""
        return self.skiplist.range_by_rank(start, end, reverse)
        
    def range_by_score(self, min_score, max_score, start=0, count=None, reverse=False):
        """Return elements with scores between min_score and max_score."""
        return self.skiplist.range_by_score(min_score, max_score, start, count, reverse)
        
    def range_by_lex(self, min_lex, max_lex, start=0, count=None, reverse=False):
        """Return elements lexicographically ordered between min_lex and max_lex."""
        return self.skiplist.range_by_lex(min_lex, max_lex, start, count, reverse)
        
    def increment(self, element, increment_by=1):
        """Increment the score of an element by the given amount."""
        current_score = self.skiplist.score(element) or 0
        new_score = current_score + increment_by
        self.skiplist.insert(element, new_score)
        return new_score
        
    # Lexicographical convenience methods
    def lexadd(self, element):
        """Add an element using lexicographical ordering only (no score)."""
        return self.add(element, self.LEX_SCORE)
        
    def lexcard(self):
        """
        Return the number of elements with the lexicographical score.
        This is an estimation as we'd need to iterate through all elements to count precisely.
        """
        # Count elements with LEX_SCORE
        count = 0
        x = self.skiplist.header
        
        # Skip to first element with LEX_SCORE
        for i in range(self.skiplist.level, -1, -1):
            while x.forward[i] and x.forward[i].score < self.LEX_SCORE:
                x = x.forward[i]
        
        # Count elements with LEX_SCORE
        x = x.forward[0]
        while x and x.score == self.LEX_SCORE:
            count += 1
            x = x.forward[0]
            
        return count
        
    def lexrange(self, min_lex, max_lex, start=0, count=None):
        """Convenience method for range_by_lex with default ascending order."""
        return self.range_by_lex(min_lex, max_lex, start, count, reverse=False)
        
    def lexrevrange(self, max_lex, min_lex, start=0, count=None):
        """Convenience method for range_by_lex with descending order."""
        return self.range_by_lex(min_lex, max_lex, start, count, reverse=True)
