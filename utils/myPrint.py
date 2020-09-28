# My utiliies
import os, sys
from io import StringIO


# Print pandas dataframe information
def printInfo(df, title=None, newLine=True, indent=2):
    buf = StringIO()
    df.info(buf=buf)
    pad_str = (' ' * indent)
    old_str = '\n'
    new_str = '\n' + pad_str
    outstr = pad_str + buf.getvalue().replace(old_str, new_str)

    if newLine: print()
    if title is not None: print(f'{title} information:')
    print(outstr)
    return


# Print pandas dataframe first N rows data
def printHead(df, num=4, title=None, newLine=True, indent=2):
    if newLine: print()
    if title is not None: print(f'{title} info and head:')
    outstr = df.head(num).to_string()
    pad_str = (' ' * indent)
    old_str = '\n'
    new_str = '\n' + pad_str
    outstr = pad_str + outstr.replace(old_str, new_str)
    print(outstr)
    return


# Pretty Print list
def printList(array, indent=2):
    isStr = isinstance(array[0], str)
    padstr = ' ' * indent
    outstr = ''
    for e in array:
        if len(outstr) > 0: outstr += ',\n'
        if isinstance(e, str):
            outstr += padstr + f"'{e}'"
        else:
            outstr += padstr + f'{e}'
    outstr = '[\n{}\n]'.format(outstr)
    print(outstr)
    return 


# Pretty print dictionary
def printDict(dic, indent=2):
    array = []
    key_maxlen = 0
    item_cnt = 0
    item_size = len(dic)
    split_str = ': '
    
    isExtLen = []
    for key, val in dic.items():
        if key_maxlen < len(str(key)): 
            key_maxlen = len(str(key))
        
        isStrKey = isinstance(key, str)
        isExtLen.append(2 if isStrKey else 1)

        tmpstr = ''
        tmpstr += f"'{key}'" if isStrKey else f"{key}"
        tmpstr += split_str
        tmpstr += f"'{val}'" if isinstance(val, str) else f"{val}"

        item_cnt += 1
        if item_cnt < item_size: tmpstr += ','
        array.append(tmpstr)

    for i in range(len(array)):
        inStr = array[i]
        ary = inStr.split(split_str)
        key = ary[0].ljust(key_maxlen + isExtLen[i])
        val = ary[1]
        array[i] = (' '*indent) + key + split_str + val
        
    outstr = '{\n' + '\n'.join(array) + '\n}'
    print(outstr)
    return


