import random
import inspect
from pynto.vocabulary import vocab

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

def print_vocab_tables():
    """
    Prints markdown tables showing all entries in vocab dict from vocabulary.py.
    Creates separate tables for each category (first element in value tuples).
    """
    # Group vocab entries by category
    categories = {}
    for word, (category, description, callable_obj) in vocab.items():
        if category not in categories:
            categories[category] = []
        categories[category].append((word, description, callable_obj))
    
    # Print each category as a separate table
    for category, entries in sorted(categories.items()):
        print(f"\n## {category}\n")
        print("| Word | Description | Parameters | Column Indexer |")
        print("|------|-------------|------------|----------------|")
        
        for word, description, callable_obj in sorted(entries):
            # Get parameter information
            try:
                # Create an instance to examine the __call__ method
                word_instance = callable_obj(word)
                call_method = getattr(word_instance, '__call__')
                sig = inspect.signature(call_method)
                
                # Extract parameters with defaults and type hints
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name in ['self', 'kwargs']:
                        continue
                    
                    # Build parameter string with type hint
                    param_str = param_name
                    if param.annotation != param.empty:
                        # Format type annotation
                        if hasattr(param.annotation, '__name__'):
                            type_hint = param.annotation.__name__
                        else:
                            type_hint = str(param.annotation)
                        # Escape pipe characters for markdown table compatibility
                        type_hint = type_hint.replace('|', '\\|')
                        param_str += f": {type_hint}"
                    
                    # Add default value if present
                    if param.default != param.empty:
                        if isinstance(param.default, str):
                            param_str += f" = '{param.default}'"
                        else:
                            param_str += f" = {param.default}"
                    
                    params.append(param_str)
                
                param_str = ", ".join(params) if params else ""
                
                # Get slice information
                slice_obj = getattr(word_instance, 'slice_', slice(None))
                if slice_obj == slice(None):
                    slice_str = "[:]"
                elif slice_obj.start is None and slice_obj.stop is None:
                    slice_str = f"[::{slice_obj.step}]" if slice_obj.step else "[:]"
                elif slice_obj.start is None:
                    slice_str = f"[:{slice_obj.stop}:{slice_obj.step}]" if slice_obj.step else f"[:{slice_obj.stop}]"
                elif slice_obj.stop is None:
                    slice_str = f"[{slice_obj.start}::{slice_obj.step}]" if slice_obj.step else f"[{slice_obj.start}:]"
                else:
                    slice_str = f"[{slice_obj.start}:{slice_obj.stop}:{slice_obj.step}]" if slice_obj.step else f"[{slice_obj.start}:{slice_obj.stop}]"
                
            except Exception as e:
                param_str = ""
                slice_str = "Error extracting slice"
            
            # Clean up the slice string to remove None values properly
            slice_str = slice_str.replace("None", "")
            
            print(f"| {word} | {description} | {param_str} | {slice_str} |")

if __name__ == '__main__':
    print_vocab_tables()
