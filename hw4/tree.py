from typing import Iterable
import numpy as np

class PreOrderIterator:
    def __init__(self, root) -> Iterable[Tree]:
        self.stack = [root]

    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.stack) == 0:
            raise StopIteration
        else:
            current = self.stack.pop()
            for c in current.children:
                self.stack.append(c)
            return current

class Tree:
    def __init__(self, name='N/A', value=None, data=None):
        if data is not None:
            self.load(data)
        else:
            self.name = name
            self.value = value
            self.children = []
            self.parent = None
            self._hidden = None
            self.depth = 0
            self.x = 0
            self.y = 0

    def is_root(self):
        return self.parent is None
    
    def set_max_depth(self, maxdepth=-1):
        if maxdepth < 0:
            self.unfold(recursive=True, predicate=lambda x: True)
        else:
            self.fold(recursive=True, predicate=lambda x: x.depth >= maxdepth)

    def load(self, data, parent=None):
        self.name = data['name']
        self.value = data['value']
        self._hidden = []
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        if 'children' in data and len(data['children']) > 0:
            _children = []
            for child in data['children']:
                _children.append(Tree().load(child, self))
            self.children = tuple(_children)
        return self

    def size(self):
        return 1 + np.sum([c.size() for c in self.children]) if len(self.children)>0 else 1

    def is_folded(self):
        return len(self._hidden) > 0

    def is_leaf(self):
        return len(self.children) == 0 and len(self._hidden) == 0

    def fold(self, recursive=False, predicate=lambda x: True):
        if recursive:
            for c in self.children:
                c.fold(recursive, predicate)
        if predicate(self) and not self.is_folded():
            # swap _hidden and children
            self.children, self._hidden = self._hidden, self.children 

    def unfold(self, recursive=False, predicate=lambda x: True):
        if recursive:
            for c in self.children:
                c.unfold(recursive, predicate)
        if self.is_folded() and predicate(self):
            self.children, self._hidden = self._hidden, self.children 

    def toggle_fold(self, recursive=False, predicate=lambda x: True):
        if recursive:
            for c in self.children:
                c.toggle_fold(recursive, predicate)
        if predicate(self):
            self.children, self._hidden = self._hidden, self.children

    def iterator(self):
        return PreOrderIterator(self)

    def __repr__(self):
        rep = ''
        for item in ['name', 'value', 'x', 'y', 'depth', 'radius', 'theta']:
            if item in self.__dict__.keys():
                rep += f'{item}: {self.__dict__[item]} ' 
        if len(self.children) > 0:
            rep += f'children: {len(self.children)}\n'
            for c in self.children:
                rep += '  ' + c.__repr__() + '\n'
        else:
            rep += 'no children\n'
        return rep

    def __str__(self):
        return self.__repr__()