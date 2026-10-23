'''
The code below implements the Buchheim et al.'s algorithm, which is 
based on Reingold-Tilford's nodelink layout algorithm. It is taken almost
verbatim from Bill Mill's repository:
https://github.com/llimllib/pymag-trees/tree/master

and specifically, the file:
https://github.com/llimllib/pymag-trees/blob/master/buchheim.py

Here is the content of the README file:

"The following repository contains implementations of the functions described 
in the article "Drawing Presentable Trees" published in Python Magazine.

Check it out at http://billmill.org/pymag-trees/

All code and text contained herein is free for anyone to use as they see fit.
It is licensed under the WTFPL, which can be found in the LICENSE file."

Please refer to the repository for additional information.

'''

__all__ = ['buchheim', 'firstwalk', 'second_walk', 'third_walk']

# augment tree with info required by layout algorithm
def convert(tree, number=1):
    tree.x = -1
    tree.y = 0
    tree.thread = None 
    tree.offset = 0
    tree.ancestor = tree
    tree.change = 0
    tree.shift = 0
    tree.number = number
    tree.mod = 0

    for i, c in enumerate(tree.children):
        convert(c, i+1)

# the following functions assume that the above `convert` function 
# has been called  already

def get_left_brother(tree):
    if tree.parent and tree.number > 1:
        return tree.parent.children[tree.number-2]
    else:
        return None

def get_leftmost_sibling(tree):
    if tree.parent and tree != tree.parent.children[0]:
        return tree.parent.children[0]
    else:
        return None

def left(tree):
    return tree.thread or len(tree.children) and tree.children[0]

def right(tree):
    return tree.thread or len(tree.children) and tree.children[-1]

# layout algorithm
def buchheim(tree): 
    convert(tree)
    dt = firstwalk(tree)
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt

def firstwalk(v, distance=1.):
    if len(v.children) == 0:
        if get_leftmost_sibling(v):
            v.x = get_left_brother(v).x + distance
        else:
            v.x = 0.
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            firstwalk(w)
            default_ancestor = apportion(w, default_ancestor,
                                         distance)
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        ell = v.children[0]
        arr = v.children[-1]
        w = get_left_brother(v)
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v

def apportion(v, default_ancestor, distance):
    w = get_left_brother(v)
    if w is not None:
        #in buchheim notation:
        #i == inner; o == outer; r == right; l == left;
        vir = vor = v
        vil = w
        vol = get_leftmost_sibling(v)
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while right(vil) and left(vir):
            vil = right(vil)
            vir = left(vir)
            vol = left(vol)
            vor = right(vor)
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                a = ancestor(vil, v, default_ancestor)
                move_subtree(a, v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if right(vil) and not right(vor):
            vor.thread = right(vil)
            vor.mod += sil - sor
        else:
            if left(vir) and not left(vol):
                vol.thread = left(vir)
                vol.mod += sir - sol
            default_ancestor = v
    return default_ancestor

def move_subtree(wl, wr, shift):
    subtrees = wr.number - wl.number
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift

def execute_shifts(v):
    shift = change = 0
    for w in v.children[::-1]:
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change

def ancestor(vil, v, default_ancestor):
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        return default_ancestor

def second_walk(v, m=0, depth=0, min=None):
    v.x += m
    v.y = depth

    if min is None or v.x < min:
        min = v.x

    for w in v.children:
        min = second_walk(w, m + v.mod, depth+1, min)

    return min

def third_walk(tree, n):
    tree.x += n
    for c in tree.children:
        third_walk(c, n)